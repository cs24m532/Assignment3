
# measure_compression.py
"""
# Computes compression ratios for weights, activations, and overall model
"""
import argparse
import torch

from models.mobilenetv2_cifar import mobilenet_v2_cifar
from compression.quant_ops import model_weight_bits, quantized_model_weight_bits
from compression.activation_wrappers import (
    wrap_activations_in_model,
    collect_activation_stats,
)
from data import get_cifar10_dataloaders
from utils import bits_to_megabytes


def main():
    # --------------------
    # Argument parsing
    # --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--activation_bits", type=int, default=8)
    parser.add_argument("--symmetric", action="store_true", default=True)
    parser.add_argument("--per_channel", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Load trained model
    # --------------------
    model = mobilenet_v2_cifar(num_classes=10, width_mult=1.0, dropout=0.2)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # --------------------
    # Weight compression
    # --------------------
    baseline_bits = model_weight_bits(model, per_tensor_bits=32)

    q_stats = quantized_model_weight_bits(
        model,
        num_bits=args.weight_bits,
        symmetric=args.symmetric,
        per_channel=args.per_channel,
    )

    print("=== Weight compression ===")
    print(f"Baseline weight bits (FP32): {baseline_bits}")
    print(f"Quantized data bits: {q_stats['data_bits']}")
    print(f"Quantized overhead bits (scale/zp): {q_stats['overhead_bits']}")
    print(f"Total quantized bits: {q_stats['total_bits']}")
    print(f"Compression ratio (weights) = baseline / quantized = "
          f"{baseline_bits / q_stats['total_bits']:.2f}x")
    print(f"Baseline weight size (MB): {bits_to_megabytes(baseline_bits):.3f}")
    print(f"Quantized weight size (MB): {bits_to_megabytes(q_stats['total_bits']):.3f}")

    # --------------------
    # Activation compression (approximate)
    # --------------------
    # Wrap activation layers to collect activation sizes
    if args.activation_bits < 32:
        wrap_activations_in_model(
            model,
            num_bits=args.activation_bits,
            symmetric=args.symmetric,
            per_channel=args.per_channel,
        )

    train_loader, _ = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
    )

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            _ = model(images)
            if i >= 10:  # sample only a few batches
                break

    # Collect per-layer activation bit usage
    act_stats = collect_activation_stats(model)
    total_act_bits = sum(act_stats.values())

    # Estimate FP32 baseline activations
    if args.activation_bits < 32 and args.activation_bits > 0:
        baseline_activation_bits_est = total_act_bits * (32 / args.activation_bits)
    else:
        baseline_activation_bits_est = total_act_bits

    print("\n=== Activation compression (approx) ===")
    print(f"Total quantized activation bits (sampled): {total_act_bits:.2e}")
    print(f"Estimated baseline (FP32) activation bits: {baseline_activation_bits_est:.2e}")
    if total_act_bits > 0:
        print(f"Compression ratio (activations) ~ "
              f"{baseline_activation_bits_est / total_act_bits:.2f}x")

    # --------------------
    # Model-level compression
    # --------------------
    baseline_model_bits = baseline_bits
    quant_model_bits = q_stats["total_bits"]

    print("\n=== Model-level compression ===")
    print(f"Model compression ratio ~ {baseline_model_bits / quant_model_bits:.2f}x")
    print(f"Final approx model size (MB) after compression: "
          f"{bits_to_megabytes(quant_model_bits):.3f} MB")


if __name__ == "__main__":
    main()
