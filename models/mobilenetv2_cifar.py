
# MobileNet-V2 implementation adapted for CIFAR-10 (32x32 images)
"""
# This file defines a lightweight convolutional neural network
# based on MobileNet-V2, modified to work well on CIFAR-10.
# The main changes vs ImageNet version are:
#   - Smaller input resolution (32x32)
#   - First convolution uses stride=1 (not 2)
#   - Number of classes = 10
"""

import torch
import torch.nn as nn
from typing import List

__all__ = ["mobilenet_v2_cifar"]


# --------------------------------------------------------------
# Utility function: make channel numbers divisible by a divisor
# --------------------------------------------------------------
# This is used in MobileNet to ensure channel counts align well
# with hardware (e.g., multiples of 8).

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Prevent rounding down too much
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# --------------------------------------------------------------
# Standard Conv-BN-ReLU6 block used throughout MobileNet-V2
# --------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            # Convolution layer
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            # Batch Normalization stabilizes training
            nn.BatchNorm2d(out_planes),
            # ReLU6 is used in MobileNet for quantization-friendliness
            nn.ReLU6(inplace=True),
        )


# --------------------------------------------------------------
# Inverted Residual Block (core of MobileNet-V2)
# --------------------------------------------------------------
# Structure:
#   1x1 expansion conv (optional)
#   3x3 depthwise conv
#   1x1 projection conv (linear)
#   Residual connection if shape matches
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride

        # Expanded channel dimension
        hidden_dim = int(round(inp * expand_ratio))

        # Residual connection only when stride=1 and channels match
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []

        # 1x1 expansion convolution
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        # 3x3 depthwise convolution
        layers.append(
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,  # depthwise
            )
        )

        # 1x1 projection convolution (no activation)
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            # Skip connection
            return x + self.conv(x)
        else:
            return self.conv(x)


# --------------------------------------------------------------
# MobileNet-V2 for CIFAR-10
# --------------------------------------------------------------
class MobileNetV2CIFAR(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        round_nearest: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # Inverted residual configuration:
        # (t = expansion factor, c = output channels,
        #  n = number of blocks, s = stride)
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # stride=1 for CIFAR (no early downsampling)
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Adjust channels using width multiplier
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )

        # First convolution layer (stride=1 for CIFAR)
        features: List[nn.Module] = [
            ConvBNReLU(3, input_channel, stride=1)
        ]

        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        # Final 1x1 convolution
        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        )

        self.features = nn.Sequential(*features)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------
# Factory function (used throughout training code)
# --------------------------------------------------------------
def mobilenet_v2_cifar(num_classes=10, width_mult=1.0, dropout=0.2):
    """
    Returns a MobileNet-V2 model configured for CIFAR-10.
    """
    return MobileNetV2CIFAR(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
    )
