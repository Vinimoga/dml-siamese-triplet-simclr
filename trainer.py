
from tqdm import tqdm

def train(model, dataloader, optimizer, epochs, device):

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

    return loss_history