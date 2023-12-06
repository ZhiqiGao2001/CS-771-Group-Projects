import os
import random

import numpy as np

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T


def build_dataset(name, split, data_folder):
    """
    Create a dataset needed for training
    """
    assert name in ["MNIST", "EMNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]
    assert split in ["train", "test"]

    # MNIST for digits
    if name == "MNIST":
        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
        ])
        dataset = datasets.MNIST(
            data_folder,
            train=(split=="train"),
            transform=transform,
            download=True
        )
        num_classes = 10
        image_shape = [1, 28, 28]

    # EMNIST for digits and letters
    elif name == "EMNIST":
        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
        ])
        dataset = datasets.EMNIST(
            data_folder,
            split="balanced",
            train=(split=="train"),
            transform=transform,
            download=True
        )
        num_classes = 47
        image_shape = [1, 28, 28]

    # CIFAR 10
    elif name == "CIFAR10":
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
        ])
        dataset = datasets.CIFAR10(
            data_folder,
            train=(split=="train"),
            transform=transform,
            download=True
        )
        num_classes = 10
        image_shape = [3, 32, 32]

    # CIFAR 100
    elif name == "CIFAR100":
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
        ])
        dataset = datasets.CIFAR100(
            data_folder,
            train=(split=="train"),
            transform=transform,
            download=True
        )
        num_classes = 100
        image_shape = [3, 32, 32]

    # Fashion MNIST (fashion images)
    elif name == "FashionMNIST":
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
        ])
        dataset = datasets.FashionMNIST(
            data_folder,
            train=(split=="train"),
            transform=transform,
            download=True
        )
        num_classes = 10
        image_shape = [1, 28, 28]

    else:
        raise ValueError('Dataset {:s} not supported.'.format(name))

    return dataset, num_classes, image_shape


def build_dataloader(dataset, batch_size, num_workers):
    """
    Create a dataloder for the target dataset
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=True,
    )
    return loader
