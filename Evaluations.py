import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

def evaluate_embeddings(model, dataloader, title="Embedding Evaluation"):
    model.eval()

    pos = []
    neg = []

    with torch.no_grad():
        for batch in tqdm(dataloader):

            if len(batch) == 3:
                # Pode ser Siamese ou Triplet

                if isinstance(batch[2], torch.Tensor) and batch[2].ndim == 1:
                    # Siamese: (x1, x2, label)
                    x1, x2, label = batch

                    z1 = model.embed(x1)
                    z2 = model.embed(x2)

                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)

                    d = F.pairwise_distance(z1, z2).cpu().numpy()

                    pos.extend(d[label == 0])
                    neg.extend(d[label == 1])

                else:
                    # Triplet: (a, p, n)
                    a, p, n = batch

                    za = model.embed(a)
                    zp = model.embed(p)
                    zn = model.embed(n)

                    za = F.normalize(za, dim=1)
                    zp = F.normalize(zp, dim=1)
                    zn = F.normalize(zn, dim=1)

                    d_ap = F.pairwise_distance(za, zp).cpu().numpy()
                    d_an = F.pairwise_distance(za, zn).cpu().numpy()

                    pos.extend(d_ap)
                    neg.extend(d_an)

            else:
                raise ValueError("Formato de batch não suportado")

    pos = np.array(pos)
    neg = np.array(neg)

    # Plot
    plt.figure()
    plt.hist(pos, bins=50, alpha=0.5, label="similar", density=True)
    plt.hist(neg, bins=50, alpha=0.5, label="different", density=True)

    # Threshold ótimo
    all_dist = np.concatenate([pos, neg])
    thresholds = np.linspace(all_dist.min(), all_dist.max(), 200)

    accs = []
    for t in thresholds:
        tp = np.sum(pos < t)
        tn = np.sum(neg > t)
        acc = (tp + tn) / (len(pos) + len(neg))
        accs.append(acc)

    best_idx = np.argmax(accs)
    best_t = thresholds[best_idx]
    best_acc = accs[best_idx]

    plt.axvline(best_t, linestyle="--", label=f"thr={best_t:.2f}")
    plt.legend()
    plt.title(title)
    plt.show()

    # Métricas
    print(f"Separation: {(neg.mean() - pos.mean()):.4f}")
    print(f"Accuracy: {best_acc:.4f}")