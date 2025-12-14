# utils.py
"""
# Utility functions shared across training,
# evaluation, and compression analysis.
"""

import os
import math
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    This ensures that:
    - Data shuffling
    - Weight initialization
    - Training results

    are repeatable across multiple runs, which is
    important for fair comparison between baseline
    and compressed models (Assignment-3 requirement).
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Safe to call even if CUDA is not available
    torch.cuda.manual_seed_all(seed)


def accuracy(output, target, topk=(1,)):
    """
    Compute Top-k accuracy.

    Args:
        output: model logits (batch_size × num_classes)
        target: ground-truth labels
        topk: tuple specifying which top-k accuracies to compute

    Returns:
        List of accuracies (in percentage) for each k in topk

    Used in:
    - Training loop (to monitor learning)
    - Evaluation (to report final test accuracy)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get indices of top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # Compare predictions with ground truth
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Count correct predictions in top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def save_checkpoint(state, filename: str):
    """
    Save model checkpoint to disk.

    The checkpoint contains:
    - Model weights
    - Optimizer state
    - Best accuracy so far
    - Training configuration

    This allows:
    - Resuming training
    - Measuring compression on trained models
      (used by measure_compression.py)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def bits_to_megabytes(bits: int) -> float:
    """
    Convert storage size from bits to megabytes (MB).

    Used in:
    - Compression analysis
    - Reporting final model size in Assignment-3

    Formula:
        bits → bytes → megabytes
    """
    return bits / 8.0 / (1024.0 * 1024.0)


def cosine_lr_scheduler(optimizer, base_lr, epoch, total_epochs):
    """
    Cosine learning-rate scheduler.

    LR(t) = 0.5 * base_lr * (1 + cos(pi * t / T))

    Used to:
    - Smoothly decay learning rate
    - Improve convergence stability
    - Match modern CNN training best practices

    Called once per epoch in train.py.
    """
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * epoch / total_epochs))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
