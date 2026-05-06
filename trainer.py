
import os
import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, epochs, device, experiment_dir=None):

    model.to(device)
    loss_history = []

    for epoch in range(epochs):
        model.train()

        for batch in tqdm(dataloader):

            batch = [b.to(device) for b in batch]

            optimizer.zero_grad()

            loss = model.training_step(batch)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        tqdm.write(f"Processed {epoch+1}/{epochs}")

        if experiment_dir is not None:
            checkpoint_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    return loss_history
