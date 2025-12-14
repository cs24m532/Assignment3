# compression/activation_wrappers.py

"""
# =====================================================================
# Activation Quantization Wrappers
# =====================================================================
# This file implements utilities to:
# 1. Wrap layers in a model so that their *activations* are fake-quantized
# 2. Collect statistics on activation bit usage
#
# These statistics are later used to compute activation compression ratios
# required by the Assignment-3 rubric.
"""

import torch
import torch.nn as nn
from typing import Dict
from .quant_ops import fake_quantize_tensor, tensor_num_bits

class QuantizedActivationWrapper(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 num_bits: int,
                 symmetric: bool = True,
                 per_channel: bool = False,
                 name: str = ""):
        super().__init__()
        self.module = module
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.name = name

        self.register_buffer("activation_bits_accum", torch.zeros(1))
        self.register_buffer("num_samples", torch.zeros(1))

    def forward(self, x):
        out = self.module(x)
        q_out = fake_quantize_tensor(
            out,
            num_bits=self.num_bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            ch_axis=1,  # channel dim for NCHW
        )
        bits = tensor_num_bits(q_out, self.num_bits)
        self.activation_bits_accum += bits
        self.num_samples += 1
        return q_out

    def get_avg_activation_bits(self) -> float:
        if self.num_samples.item() == 0:
            return 0.0
        return self.activation_bits_accum.item() / self.num_samples.item()

def wrap_activations_in_model(model: nn.Module,
                              num_bits: int,
                              symmetric: bool = True,
                              per_channel: bool = False):
    for name, child in list(model.named_children()):
        if isinstance(child, (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6, nn.Sequential)):
            wrapped = QuantizedActivationWrapper(
                module=child,
                num_bits=num_bits,
                symmetric=symmetric,
                per_channel=per_channel,
                name=name,
            )
            setattr(model, name, wrapped)
        else:
            wrap_activations_in_model(child, num_bits, symmetric, per_channel)

def collect_activation_stats(model: nn.Module) -> Dict[str, float]:
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedActivationWrapper):
            stats[name] = module.get_avg_activation_bits()
    return stats
