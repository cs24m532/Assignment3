## README — Reproducibility and Execution Details

### Environment Setup

- **OS:** Ubuntu 20.04  
- **Python:** 3.9.18 (Anaconda)  
- **CUDA:** 11.8  
- **GPU:** NVIDIA RTX series (CPU fallback supported)

---

### Conda Environment Setup

```bash
conda create -n cs6886w python=3.9 -y
conda activate cs6886w

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

Implemented via:

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


This run reports:

Baseline top-1 accuracy

Loss and accuracy curves

Reference FP32 model size

Compression-Aware Training Runs (Minimum 8 Runs)

The following configurations were used to generate the WandB Parallel Coordinates plot:

python train.py --weight_bits 8 --activation_bits 8 --use_wandb
python train.py --weight_bits 8 --activation_bits 6 --use_wandb
python train.py --weight_bits 6 --activation_bits 8 --use_wandb
python train.py --weight_bits 6 --activation_bits 6 --use_wandb
python train.py --weight_bits 4 --activation_bits 8 --use_wandb
python train.py --weight_bits 8 --activation_bits 4 --use_wandb
python train.py --weight_bits 6 --activation_bits 4 --use_wandb
python train.py --weight_bits 4 --activation_bits 6 --use_wandb


✔ Minimum 8 compression simulations satisfied
✔ Weight–activation mixed precision explored

Compression Measurement
python measure_compression.py \
  --checkpoint checkpoints/checkpoint_epoch_199.pth \
  --weight_bits 6 \
  --activation_bits 6


This script reports:

Weight compression ratio

Activation compression ratio

Model-level compression ratio

Final compressed model size (MB)

Storage overheads (scales, zero-points)

WandB Parallel Coordinates Plot

Steps to generate the plot:

Open the WandB project: cs6886_assignment3

Select all compression runs

Click Charts → Parallel Coordinates

Configure axes as:

weight_bits

activation_bits

val/acc1

model_compression_ratio

This visualization is used to analyze accuracy vs. compression trade-offs.

Summary

Evaluation is embedded directly in train.py

eval.py is intentionally unused

Compression analysis is isolated in measure_compression.py

All experiments are reproducible with fixed seeds
