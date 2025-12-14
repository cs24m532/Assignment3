## README — Reproducibility and Execution Details

This section documents the exact environment, commands, dependencies, and seed configuration used to reproduce all results reported in Assignment-3.

---

## Environment Setup

- **Operating System**: Ubuntu 20.04  
- **Python**: 3.9.18  
- **CUDA**: 11.8  
- **GPU**: NVIDIA RTX series (CPU fallback supported)

---

## Conda Environment Creation

All experiments were run inside a dedicated Conda environment.

```bash
conda create -n cs6886_a3 python=3.9 -y
conda activate cs6886_a3
```

## Python Dependencies
Exact dependency versions used:

```bash
pip install torch==2.1.0 torchvision==0.16.0
pip install numpy==1.24.4
pip install wandb==0.16.3
```

## Seed Configuration (Reproducibility)

All experiments use a fixed random seed:

seed = 42

##Baseline Training (No Compression)

```bash
python train.py \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.1 \
  --weight_bits 32 \
  --activation_bits 32 \
  --use_wandb
```

This run reports:

Baseline Top-1 accuracy
Training and validation loss/accuracy curves


## Compression-Aware Training Runs (Minimum 8 Runs)
The following configurations were used to generate the WandB Parallel Coordinates plot.

```bash

python train.py --weight_bits 8 --activation_bits 8 --use_wandb
python train.py --weight_bits 8 --activation_bits 6 --use_wandb
python train.py --weight_bits 6 --activation_bits 8 --use_wandb
python train.py --weight_bits 6 --activation_bits 6 --use_wandb
python train.py --weight_bits 4 --activation_bits 8 --use_wandb
python train.py --weight_bits 8 --activation_bits 4 --use_wandb
python train.py --weight_bits 6 --activation_bits 4 --use_wandb
python train.py --weight_bits 4 --activation_bits 6 --use_wandb

```
## Compression Measurement
Compression statistics are measured using a trained checkpoint:

```bash

python measure_compression.py \
  --checkpoint checkpoints/checkpoint_epoch_199.pth \
  --weight_bits 6 \
  --activation_bits 6

```
This script reports:

Weight compression ratio
Activation compression ratio
Model compression ratio
Storage overheads (scales, zero-points)
Final approximated compressed model size (MB)

## WandB Parallel Coordinates Plot

Steps to generate the plot:

Open the WandB project: cs6886_assignment3
Select all compression runs
Click Charts → Parallel Coordinates

Use the following axes:

weight_bits
activation_bits
val/acc1
model_compression_ratio
