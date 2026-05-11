import torch
import torchvision
import os
import json
import sys
import glob
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import SiameseDataset, TripletDataset, SimCLRDataset, SupConDataset
from models import MLP, LeNet, AlexNet, TripletNetwork, SiameseNetwork, SimCLRNetwork, SupConNetwork
from loss import ContrastiveLoss
from Evaluations import evaluate_embeddings, get_embeddings, get_random_samples, plot_tsne, evaluate_original_embeddings
from utils import load_config, get_base_dataset

def build_model(config):
    #device
    device = config["device"].lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Suporte para AMD (DirectML)
    if device.lower() in ["dml", "amd"]:
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"Usando GPU AMD via DirectML: {device}")
        except ImportError:
            device = torch.device("cpu")

    #backbone
    backbone_name = config["backbone"]
    image_channels = config["image_channels"]
    embedding_dim = config["embedding_dim"]
    
    if backbone_name == "LeNet":
        backbone = LeNet(input_dim=image_channels, output_dim=embedding_dim)
    elif backbone_name == "AlexNet":
        backbone = AlexNet(input_dim=image_channels, output_dim=embedding_dim)
    else:
        raise ValueError(f"Backbone inválido: {backbone_name}")

    #wrapped dataset, loss and model
    model_type = config["model"].lower()
    
    if model_type == "siamese":
        criterion = ContrastiveLoss(margin=config.get("margin", 1.0))
        model = SiameseNetwork(backbone, criterion)
        
    elif model_type == "triplet":
        criterion = torch.nn.TripletMarginLoss(margin=config.get("margin", 1.0))
        model = TripletNetwork(backbone, criterion)
        
    elif model_type == "simclr":
        model = SimCLRNetwork(
            backbone, 
            projection_dim=config.get("projection_dim", 64), 
            temperature=config.get("temperature", 0.5)
        )
        
    elif model_type == "supcon":
        model = SupConNetwork(
            backbone, 
            projection_dim=config.get("projection_dim", 64), 
            temperature=config.get("temperature", 0.5)
        )
    else:
        raise ValueError(f"Modelo inválido: {model_type}")
        
    return model, device

def evaluate_experiment(exp_dir):
    print(f"=======================================")
    print(f"Evaluating Experiment in {exp_dir}")
    print(f"=======================================")
    
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"No config.json found in {exp_dir}. Skipping.")
        return
        
    config = load_config(config_path)
    
    #dataset
    train_dataset, test_dataset = get_base_dataset(config["dataset"])
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )
    
    epochs = config["epochs"]
    
    results = {
        "config": config,
        "epochs_eval": {}
    }
    
    # get 200 random samples from test dataset for tsne
    all_data = []
    all_labels = []
    print("Loading test dataset for random sampling...")
    for data, label in test_dataset:
        all_data.append(data.numpy())
        all_labels.append(label)
    
    import numpy as np
    sampled_data, sampled_labels = get_random_samples(all_data, all_labels, samples_per_class=200)
    
    # Convert back to tensor and dataloader for embedding extraction
    sampled_dataset = torch.utils.data.TensorDataset(torch.tensor(sampled_data), torch.tensor(sampled_labels))
    sampled_dataloader = DataLoader(sampled_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Create results directories
    results_dir = os.path.join(exp_dir, "results")
    tsne_dir = os.path.join(results_dir, "tsne")
    dist_dir = os.path.join(results_dir, "distributions")
    os.makedirs(tsne_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)
    
    # 4. Fazer a loss curve
    loss_path = os.path.join(exp_dir, "loss_curve.csv")
    if os.path.exists(loss_path):
        loss_df = pd.read_csv(loss_path)
        plt.figure(figsize=(12, 5))
        if "epoch" in loss_df.columns and "loss" in loss_df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            
            # Plot raw loss over batches
            ax1.plot(loss_df.index, loss_df["loss"], alpha=0.5, color='orange')
            ax1.set_title("Loss vs Batches")
            ax1.set_xlabel("Batch (Global)")
            ax1.set_ylabel("Loss")
            ax1.set_ylim(bottom=0)
            
            # Plot average loss per epoch
            epoch_loss = loss_df.groupby("epoch")["loss"].mean()
            ax2.plot(epoch_loss.index, epoch_loss.values, marker='o')
            ax2.set_title("Average Loss vs Epoch")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Mean Loss")
            ax2.set_ylim(bottom=0)
            
            plt.tight_layout()
        else:
            # Fallback if columns are different
            plt.figure(figsize=(6, 5))
            plt.plot(loss_df.index, loss_df.iloc[:, -1])
            plt.title("Loss Curve")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.ylim(bottom=0)
        plt.savefig(os.path.join(results_dir, "loss_curve.png"))
        plt.close()
        print("Loss curve saved.")
    
    for epoch in range(0, epochs + 1):
        print(f"\n--- Evaluating Epoch {epoch} ---")
        model, device = build_model(config)
        
        if epoch > 0:
            model_path = os.path.join(exp_dir, f"model_epoch_{epoch}.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"Model for epoch {epoch} not found at {model_path}. Skipping.")
                continue
        else:
            print("Using untrained model for Epoch 0.")
                
        model.to(device)
        model.eval()
        
        # 1 and 2: Distribution for this epoch
        print(f"Calculating distribution for epoch {epoch}...")
        separation, best_acc = evaluate_embeddings(model, test_dataloader, title=f"Embedding Distribution - Epoch {epoch}", device=device, file_path=dist_dir)
        
        # 3: Extract embeddings for the 200 samples and t-SNE
        print(f"Extracting embeddings for t-SNE at epoch {epoch}...")
        embeddings, labels = get_embeddings(sampled_dataloader, model, device)
        
        # also apply original embedding evaluation
        orig_metrics = evaluate_original_embeddings(embeddings, labels)
        
        tsne_title = f"t-SNE_epoch_{epoch}"
        data_2d, tsne_metrics = plot_tsne(embeddings, labels, title=tsne_title, file_path=tsne_dir)
        
        results["epochs_eval"][f"epoch_{epoch}"] = {
            "separation": float(separation),
            "balanced_accuracy": float(best_acc),
            "original_embedding_metrics": orig_metrics,
            "tsne_metrics": tsne_metrics
        }
        
    # 5. Fazer arquivo final json
    with open(os.path.join(results_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFinished evaluation for {exp_dir}. Results saved.")


def main():
    if len(sys.argv) < 2:
        print("Uso: python evaluate.py <caminho_para_pasta_experimentos>")
        sys.exit(1)

    base_path = sys.argv[1]
    
    if not os.path.isdir(base_path):
        print(f"Erro: {base_path} não é um diretório.")
        sys.exit(1)
        
    # Verifica se a pasta já é um experimento (tem config.json) ou se contém experimentos
    if os.path.exists(os.path.join(base_path, "config.json")):
        evaluate_experiment(base_path)
    else:
        # Percorrer todas as subpastas
        for item in os.listdir(base_path):
            sub_path = os.path.join(base_path, item)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "config.json")):
                evaluate_experiment(sub_path)

if __name__ == "__main__":
    main()