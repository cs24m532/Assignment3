Environment Setup

OS: Ubuntu 20.04

Python: 3.9.18

CUDA: 11.8

GPU: NVIDIA RTX series (or CPU fallback supported)

Python Dependencies
pip install torch==2.1.0 torchvision==0.16.0
pip install numpy==1.24.4
pip install wandb==0.16.3

Seed Configuration (Reproducibility)

All experiments are run with a fixed seed:

seed = 42


The seed is applied to:

Python random

NumPy

PyTorch (CPU + CUDA)

set_seed(42)


This ensures reproducible training, evaluation, and compression measurements.

Baseline Training (No Compression)
python train.py \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.1 \
  --weight_bits 32 \
  --activation_bits 32 \
  --use_wandb

Compression-Aware Training Runs (≥ 8 runs)

Example configurations used for WandB Parallel Coordinates:

python train.py --weight_bits 8 --activation_bits 8 --use_wandb
python train.py --weight_bits 8 --activation_bits 6 --use_wandb
python train.py --weight_bits 6 --activation_bits 8 --use_wandb
python train.py --weight_bits 6 --activation_bits 6 --use_wandb
python train.py --weight_bits 4 --activation_bits 8 --use_wandb
python train.py --weight_bits 8 --activation_bits 4 --use_wandb
python train.py --weight_bits 6 --activation_bits 4 --use_wandb
python train.py --weight_bits 4 --activation_bits 6 --use_wandb


✔ Minimum 8 runs satisfied

Compression Measurement
python measure_compression.py \
  --checkpoint checkpoints/checkpoint_epoch_199.pth \
  --weight_bits 6 \
  --activation_bits 6


This reports:

Weight compression ratio

Activation compression ratio

Model compression ratio

Final compressed model size (MB)

Storage overheads (scales, zero-points)

WandB Parallel Coordinates Plot

Steps:

Open WandB project cs6886_assignment3

Select all compression runs

Click Charts → Parallel Coordinates

Axes used:

weight_bits

activation_bits

val/acc1

model_compression_ratio

Summary

Evaluation is embedded in train.py

eval.py is intentionally unused

Compression analysis is isolated in measure_compression.py

All results are reproducible with fixed seeds
