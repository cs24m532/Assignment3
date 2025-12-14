
# train.py
"""
# This script trains MobileNet-V2 on CIFAR-10 with optional
# fake quantization of weights and activations, and logs results to WandB.
"""
import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainConfig
from data import get_cifar10_dataloaders
from utils import set_seed, accuracy, save_checkpoint, cosine_lr_scheduler
from models.mobilenetv2_cifar import mobilenet_v2_cifar
from compression.quant_ops import fake_quantize_tensor
from compression.activation_wrappers import wrap_activations_in_model

# Optional WandB import (script works even if wandb is not installed)
try:
    import wandb
except ImportError:
    wandb = None


# -------------------------------------------------
# Parse CLI arguments and map them to TrainConfig
# -------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    # -------- Data & model --------
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)

    # -------- Optimizer --------
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_schedule", type=str, default="cosine")

    # -------- Compression --------
    parser.add_argument("--weight_bits", type=int, default=32)
    parser.add_argument("--activation_bits", type=int, default=32)
    parser.add_argument("--symmetric", action="store_true", default=True)
    parser.add_argument("--per_channel", action="store_true", default=False)
    parser.add_argument("--quantize_weights_during_forward", action="store_true")
    parser.add_argument("--quantize_activations_during_train", action="store_true")

    # -------- Logging --------
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs6886_assignment3")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Convert argparse args into TrainConfig dataclass
    cfg = TrainConfig(
        data_dir=args.data_dir,
        width_mult=args.width_mult,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        symmetric=args.symmetric,
        per_channel=args.per_channel,
        quantize_weights_during_forward=args.quantize_weights_during_forward,
        quantize_activations_during_train=args.quantize_activations_during_train,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_interval=args.log_interval,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    return cfg


# -------------------------------------------------
# Main training pipeline
# -------------------------------------------------

def main():
    cfg = parse_args()
    print(">>> main() started with config:", cfg)

    # Ensure reproducibility
    set_seed(cfg.seed)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)

    # Load CIFAR-10 data
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(">>> CIFAR-10 dataloaders ready")

    # Create MobileNet-V2 model (CIFAR-10 variant)
    model = mobilenet_v2_cifar(
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    ).to(device)
    print(">>> Model created")

    # Optionally wrap activation layers with fake-quant modules
    if cfg.quantize_activations_during_train and cfg.activation_bits < 32:
        wrap_activations_in_model(
            model,
            num_bits=cfg.activation_bits,
            symmetric=cfg.symmetric,
            per_channel=cfg.per_channel,
        )
        print(">>> Activation quantization wrappers enabled")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer: SGD with momentum and Nesterov acceleration
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    # Initialize WandB logging if enabled
    if cfg.use_wandb and wandb is not None:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.as_dict(),
        )
        print(">>> WandB logging enabled")

    best_acc1 = 0.0
    print(">>> Starting training loop...")

    # ---------------- Training loop ----------------
    for epoch in range(cfg.epochs):
        # Update learning rate
        if cfg.lr_schedule == "cosine":
            lr = cosine_lr_scheduler(optimizer, cfg.lr, epoch, cfg.epochs)
        else:
            lr = optimizer.param_groups[0]["lr"]

        # Train for one epoch
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg)

        # Evaluate on test set
        acc1, acc5 = evaluate(model, test_loader, criterion, device)

        # Log metrics to WandB
        if cfg.use_wandb and wandb is not None:
            wandb.log({
                "val/acc1": acc1,
                "val/acc5": acc5,
                "epoch": epoch,
                "lr": lr,
            })

        # Track best accuracy
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # Save checkpoint every epoch
        os.makedirs(cfg.out_dir, exist_ok=True)
        save_path = os.path.join(cfg.out_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "config": cfg.as_dict(),
            },
            save_path,
        )

    # Finish WandB run
    if cfg.use_wandb and wandb is not None:
        wandb.finish()


# -------------------------------------------------
# Fake quantization of weights during forward pass
# -------------------------------------------------

def maybe_fake_quantize_weights(model: nn.Module, cfg: TrainConfig):
    """
    Fake-quantize Conv2d and Linear weights before each forward pass.
    This simulates low-bit inference during training.
    """
    if cfg.weight_bits >= 32 or not cfg.quantize_weights_during_forward:
        return

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                q_w = fake_quantize_tensor(
                    m.weight,
                    num_bits=cfg.weight_bits,
                    symmetric=cfg.symmetric,
                    per_channel=cfg.per_channel,
                    ch_axis=0,
                )
                m.weight.copy_(q_w)


# -------------------------------------------------
# Train one epoch
# -------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    cfg: TrainConfig,
):
    model.train()

    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    total_samples = 0

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Optional fake weight quantization
        maybe_fake_quantize_weights(model, cfg)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        running_acc1 += acc1.item() * batch_size
        running_acc5 += acc5.item() * batch_size
        total_samples += batch_size

        # Periodic logging
        if (i + 1) % cfg.log_interval == 0:
            avg_loss = running_loss / total_samples
            avg_acc1 = running_acc1 / total_samples
            print(
                f"Epoch [{epoch}] Step [{i+1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f} Acc@1: {avg_acc1:.2f}"
            )


# -------------------------------------------------
# Evaluation on validation/test set
# -------------------------------------------------

def evaluate(
    model: nn.Module,
    data_loader,
    criterion,
    device,
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)

            running_loss += loss.item() * batch_size
            running_acc1 += acc1.item() * batch_size
            running_acc5 += acc5.item() * batch_size
            total_samples += batch_size

    avg_acc1 = running_acc1 / total_samples
    avg_acc5 = running_acc5 / total_samples
    avg_loss = running_loss / total_samples

    print(
        f"Eval: Loss {avg_loss:.4f} Acc@1 {avg_acc1:.2f} Acc@5 {avg_acc5:.2f}"
    )
    return avg_acc1, avg_acc5


if __name__ == "__main__":
    main()
