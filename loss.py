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
    """
    NT-Xent Loss for Contrastive Learning
    based on:
    https://towardsdatascience.com/nt-xent-normalized-temperature-scaled-cross-entropy-loss-explained-and-implemented-in-pytorch-cc081f69848/
    https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
    
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)

        sim_matrix = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, -9e15) #sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        positives = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ])

        exp_sim = torch.exp(sim_matrix)

        denom = exp_sim.sum(dim=1)

        loss = -torch.log(torch.exp(positives) / denom)

        loss = loss.mean()

        return loss
        
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    Based on https://arxiv.org/pdf/2004.11362.pdf
    """
    def __init__(self, temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # features is expected to be of shape [batch_size, n_views, embed_dim]
        # For SupCon, usually n_views = 2 (x1 and x2)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        mask = mask.repeat(anchor_count, contrast_count)
        
        logits = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        # +1e-9 to prevent log(0)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_sum = mask.sum(1)
        # Avoid division by zero if a class only has 1 sample in the batch
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss
