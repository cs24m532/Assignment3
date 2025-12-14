# compression/quant_ops.py
import torch
import torch.nn as nn
from typing import Dict

def quantize_tensor(x: torch.Tensor,
                    num_bits: int,
                    symmetric: bool = True,
                    per_channel: bool = False,
                    ch_axis: int = 0):
    if num_bits >= 32:
        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        zero_point = torch.zeros(1, device=x.device, dtype=torch.int32)
        return x.to(torch.int32), scale, zero_point

    qmin = 0
    qmax = 2 ** num_bits - 1

    if per_channel and x.dim() > 1:
        reduce_dims = [d for d in range(x.dim()) if d != ch_axis]
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()

    if symmetric:
        max_abs = torch.max(x_min.abs(), x_max.abs())
        x_min = -max_abs
        x_max = max_abs

    eps = 1e-8
    scale = (x_max - x_min).clamp(min=eps) / float(qmax - qmin)
    zero_point = qmin - torch.round(x_min / scale)
    zero_point = zero_point.clamp(qmin, qmax).to(torch.int32)

    q_x = torch.round(x / scale + zero_point)
    q_x = q_x.clamp(qmin, qmax).to(torch.int32)
    return q_x, scale, zero_point

def dequantize_tensor(q_x: torch.Tensor,
                      scale: torch.Tensor,
                      zero_point: torch.Tensor) -> torch.Tensor:
    return (q_x.to(scale.dtype) - zero_point.to(scale.dtype)) * scale

def fake_quantize_tensor(x: torch.Tensor,
                         num_bits: int,
                         symmetric: bool = True,
                         per_channel: bool = False,
                         ch_axis: int = 0) -> torch.Tensor:
    q_x, scale, zero_point = quantize_tensor(
        x,
        num_bits=num_bits,
        symmetric=symmetric,
        per_channel=per_channel,
        ch_axis=ch_axis,
    )
    return dequantize_tensor(q_x, scale, zero_point)

def tensor_num_bits(x: torch.Tensor, num_bits: int) -> int:
    return x.numel() * num_bits

def overhead_for_scale_zeropoint(scale: torch.Tensor,
                                 zero_point: torch.Tensor) -> int:
    num_scales = scale.numel()
    num_zp = zero_point.numel()
    return (num_scales + num_zp) * 32

def model_weight_bits(model: nn.Module,
                      per_tensor_bits: int = 32) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * per_tensor_bits
    return total

def quantized_model_weight_bits(model: nn.Module,
                                num_bits: int,
                                symmetric: bool = True,
                                per_channel: bool = False) -> Dict[str, int]:
    total_weight_bits = 0
    total_overhead_bits = 0

    for _, p in model.named_parameters():
        x = p.detach().cpu()
        if per_channel and x.dim() > 1:
            ch_axis = 0
        else:
            ch_axis = 0
        q_x, scale, zp = quantize_tensor(
            x, num_bits=num_bits, symmetric=symmetric,
            per_channel=per_channel, ch_axis=ch_axis
        )
        total_weight_bits += tensor_num_bits(q_x, num_bits)
        total_overhead_bits += overhead_for_scale_zeropoint(scale, zp)

    return {
        "data_bits": total_weight_bits,
        "overhead_bits": total_overhead_bits,
        "total_bits": total_weight_bits + total_overhead_bits,
    }
