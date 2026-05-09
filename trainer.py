import os
import torch
import time
import csv
import json
from tqdm import tqdm

def train(model, dataloader, optimizer, epochs, device, experiment_dir=None):
    model.to(device)
    loss_history = []
    batch_metrics = []
    
    total_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch_start = time.time()
            
            batch = [b.to(device) for b in batch]

            optimizer.zero_grad()

            loss = model.training_step(batch)

            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start
            loss_val = loss.item()
            loss_history.append(loss_val)
            epoch_loss += loss_val
            
            batch_metrics.append({
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
                "loss": loss_val,
                "batch_time_sec": batch_time,
                "iter_per_sec": 1.0 / batch_time if batch_time > 0 else 0
            })

        avg_epoch_loss = epoch_loss / len(dataloader)
        tqdm.write(f"Epoch {epoch+1} Completed | Avg Loss: {avg_epoch_loss:.4f}")

        if experiment_dir is not None:
            checkpoint_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    total_time = time.time() - total_start_time
    total_iters = len(loss_history)
    avg_iter_per_sec = total_iters / total_time if total_time > 0 else 0
    
    if experiment_dir is not None:
        # Save CSV
        csv_path = os.path.join(experiment_dir, "loss_curve.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "batch", "loss", "batch_time_sec", "iter_per_sec"])
            writer.writeheader()
            writer.writerows(batch_metrics)
            
        # Save Stats
        stats = {
            "total_time_sec": total_time,
            "total_epochs": epochs,
            "total_batches": total_iters,
            "avg_iter_per_sec": avg_iter_per_sec,
            "final_loss": loss_history[-1] if loss_history else None
        }
        with open(os.path.join(experiment_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)
            
        print(f"\nTreinamento concluído em {total_time:.2f}s ({avg_iter_per_sec:.2f} iters/s).")
        print(f"Métricas (CSV de loss e stats) salvas em: {experiment_dir}")

    return loss_history
