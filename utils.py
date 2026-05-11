import torchvision
import torch
import torchvision.transforms as transforms

import os
from datetime import datetime

import json

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
