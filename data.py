# data.py
"""
CIFAR-10 data loading and preprocessing utilities.

Includes:
- Proper normalization using dataset mean/std
- Data augmentation for training set
- Clean separation between train and test pipelines
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def get_cifar10_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
):
    """
    Creates CIFAR-10 train and test dataloaders with standard preprocessing.

    Args:
        data_dir (str): Directory to download/load CIFAR-10
        batch_size (int): Batch size for training/testing
        num_workers (int): Number of DataLoader workers

    Returns:
        train_loader, test_loader
    """

    # CIFAR-10 channel-wise mean and standard deviation
    # Computed over the training set
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # --------------------
    # Training transforms
    # --------------------
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),      # Data augmentation
        T.RandomHorizontalFlip(),         # Improves generalization
        T.ToTensor(),                     # Convert PIL image to tensor
        T.Normalize(mean, std),            # Normalize inputs
    ])

    # --------------------
    # Test transforms
    # --------------------
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # CIFAR-10 test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,   # Useful when CUDA is available
    )

    # Test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
