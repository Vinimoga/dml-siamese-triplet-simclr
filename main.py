import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import sys
import os
from datetime import datetime

from datasets import SiameseDataset, TripletDataset, SimCLRDataset, SupConDataset
from models import MLP, LeNet, AlexNet, TripletNetwork, SiameseNetwork, SimCLRNetwork, SupConNetwork
from loss import ContrastiveLoss
from trainer import train
from Evaluations import evaluate_embeddings

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


def run_experiment(config_path):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iniciando experimento usando: {config_path}")

    #configs
    config = load_config(config_path)

    #device
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    #dataset
    train_dataset, test_dataset = get_base_dataset(config["dataset"])
    image_channels = config["image_channels"]
    embedding_dim = config["embedding_dim"]
    
    #backbone
    backbone_name = config["backbone"]
    
    if backbone_name == "LeNet":
        backbone = LeNet(input_dim=image_channels, output_dim=embedding_dim)
    elif backbone_name == "AlexNet":
        backbone = AlexNet(input_dim=image_channels, output_dim=embedding_dim)
    else:
        raise ValueError(f"Backbone inválido: {backbone_name}")

    #wrapped dataset, loss and model
    model_type = config["model"].lower()
    
    if model_type == "siamese":
        train_wrapped = SiameseDataset(train_dataset)
        criterion = ContrastiveLoss(margin=config.get("margin", 1.0))
        model = SiameseNetwork(backbone, criterion)
        
    elif model_type == "triplet":
        train_wrapped = TripletDataset(train_dataset)
        criterion = torch.nn.TripletMarginLoss(margin=config.get("margin", 1.0))
        model = TripletNetwork(backbone, criterion)
        
    elif model_type == "simclr":
        train_wrapped = SimCLRDataset(train_dataset)
        model = SimCLRNetwork(
            backbone, 
            projection_dim=config.get("projection_dim", 64), 
            temperature=config.get("temperature", 0.5)
        )
        
    elif model_type == "supcon":
        train_wrapped = SupConDataset(train_dataset)
        model = SupConNetwork(
            backbone, 
            projection_dim=config.get("projection_dim", 64), 
            temperature=config.get("temperature", 0.5)
        )
    else:
        raise ValueError(f"Modelo inválido: {model_type}")
    
    #dataloaders
    train_dataloader = DataLoader(
        train_wrapped,
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    #optimizer
    optimizer_name = config.get("optimizer", "adam").lower()
    lr = config.get("lr", 0.001)
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    #experiment
    experiment_dir = f"experiments/{model.__class__.__name__}/{backbone.__class__.__name__}_{config['dataset']}_{embedding_dim}_{lr}_{config['epochs']}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    #save configs
    with open(f"{experiment_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    #train
    print(f"Iniciando treinamento do modelo {model_type} com backbone {backbone_name}...")
    train(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=config["epochs"],
        device=device,
        experiment_dir=experiment_dir
    )

    # #save
    # save_path = config.get("save_path", "models/")
    # os.makedirs(save_path, exist_ok=True)
    # torch.save(model.backbone.state_dict(), f"{save_path}{model_type}_backbone.pth")
    # torch.save(model.state_dict(), f"{experiment_dir}/model.pth")
    
    print(f"Experimento concluído. Resultados salvos em: {experiment_dir}\n")

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py configs/config.json ou python main.py configs/")
        sys.exit(1)

    path = sys.argv[1]
    
    if os.path.isdir(path):
        config_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")]
    else:
        config_files = [path]
        
    for cf in config_files:
        run_experiment(cf)

if __name__ == "__main__":
    main()