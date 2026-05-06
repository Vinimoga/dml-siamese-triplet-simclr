import json
import os
import itertools

def generate_grid():
    datasets = [("MNIST", 1), ("CIFAR10", 3)]
    lrs = [1e-3, 1e-4]
    backbones = ["AlexNet", "LeNet"]
    epochs = 5
    model_type = "siamese"

    os.makedirs(f"configs/{model_type}", exist_ok=True)

    combinations = list(itertools.product(datasets, lrs, backbones))
    
    for i, ((dataset, channels), lr, backbone) in enumerate(combinations):
        config = {
            "dataset": dataset,
            "image_channels": channels,
            "model": model_type,
            "backbone": backbone,
            "batch_size": 64,
            "epochs": epochs,
            "lr": lr,
            "embedding_dim": 128,
            "margin": 1.0,
            "optimizer": "adam",
            "device": "cuda",
            "save_path": "models/"
        }
        
        filename = f"configs/{model_type}/exp_{dataset}_{backbone}_lr{lr}.json"
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
            
    print(f"Configs generated at configs/{model_type}")

if __name__ == "__main__":
    generate_grid()
