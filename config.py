# config.py
"""
Configuration file for CS6886W Assignment-3
-------------------------------------------

This file defines all training, model, compression, and logging
hyperparameters in a single dataclass. The same configuration
is used consistently across baseline and compressed experiments
to ensure fair comparison.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    # =========================
    # Dataset & DataLoader
    # =========================

    # Root directory where CIFAR-10 is downloaded/stored
    data_dir: str = "./data"

    # Number of worker processes for data loading
    # (kept small since experiments were run on CPU)
    num_workers: int = 4


    # =========================
    # Model Configuration
    # =========================

    # Width multiplier for MobileNet-V2
    # width_mult = 1.0 corresponds to standard MobileNet-V2
    width_mult: float = 1.0

    # Dropout probability used in classifier head
    # Helps reduce overfitting
    dropout: float = 0.2

    # Number of output classes (CIFAR-10)
    num_classes: int = 10


    # =========================
    # Training Hyperparameters
    # =========================

    # Mini-batch size
    batch_size: int = 128

    # Total number of training epochs
    # (Baseline trained longer, quantized runs trained fewer epochs)
    epochs: int = 200

    # Optimizer type: SGD used for all experiments
    optimizer: str = "sgd"

    # Initial learning rate
    lr: float = 0.1

    # Momentum for SGD optimizer
    momentum: float = 0.9

    # L2 regularization (weight decay)
    weight_decay: float = 5e-4

    # Learning rate schedule:
    # "cosine" → cosine annealing with warmup
    lr_schedule: str = "cosine"

    # Number of warmup epochs before cosine decay
    warmup_epochs: int = 5

    # Label smoothing (not used in final experiments)
    label_smoothing: float = 0.0

    # Random seed for reproducibility
    seed: int = 42


    # =========================
    # Compression Parameters
    # =========================

    # Bit-width used to quantize model weights
    # Examples tested: 32 (baseline), 8, 6, 4, 2
    weight_bits: int = 8

    # Bit-width used to quantize activations
    # Same values as weight_bits explored
    activation_bits: int = 8

    # Symmetric quantization flag
    # True → zero-point = 0
    symmetric: bool = True

    # Per-channel quantization (disabled to keep analysis simple)
    per_channel: bool = False

    # Whether to quantize activations during training
    # (kept False → post-training quantization)
    quantize_activations_during_train: bool = False

    # Whether to quantize weights during forward pass
    # (kept False → quantization only for analysis)
    quantize_weights_during_forward: bool = False


    # =========================
    # Logging & Output
    # =========================

    # Enable Weights & Biases logging
    use_wandb: bool = False

    # WandB project name
    wandb_project: str = "cs6886_assignment3"

    # Optional run name (auto-generated if None)
    wandb_run_name: Optional[str] = None

    # Interval (in steps) for printing training logs
    log_interval: int = 100

    # Directory where checkpoints are saved
    out_dir: str = "./checkpoints"


    # =========================
    # Utility
    # =========================

    def as_dict(self):
        """
        Converts configuration to dictionary format.
        Used for logging (e.g., WandB config tracking).
        """
        return dataclasses.asdict(self)
