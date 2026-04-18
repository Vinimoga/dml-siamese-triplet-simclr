import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calcula a distância euclidiana entre os embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Loss: se label=0 (similar), minimiza dist; se label=1 (diferente), aumenta dist
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)

        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # positivos: i com i+batch_size
        positives = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ])

        labels = torch.arange(2 * batch_size).to(z.device)
        labels = (labels + batch_size) % (2 * batch_size)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss