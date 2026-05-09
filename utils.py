import torchvision
import torch
import torchvision.transforms as transforms

import os
from datetime import datetime

import json


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def get_base_dataset(name):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if name == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError("Dataset não suportado")

    return train_dataset, test_dataset
