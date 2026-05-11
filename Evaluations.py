import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, confusion_matrix
from sklearn.manifold import TSNE
def evaluate_embeddings(model, dataloader, title="Embedding Evaluation", device="cpu", file_path="."):
    model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Tenta descobrir de forma inteligente o que o dataloader está retornando
            if len(batch) == 2 and batch[1].ndim == 1:
                # Dataloader Padrão (x, label_real)
                x, y = batch
            elif len(batch) == 3 and batch[2].ndim == 1 and batch[2].max() > 1:
                # SupCon Dataset (x1, x2, label_real)
                x, _, y = batch
            else:
                # Se for Siamese, Triplet ou SimCLR puro, apenas pegamos o primeiro item (x)
                # Isso não é o ideal para a curva de distribuição, por isso vou avisar abaixo
                print("AVISO: Para a melhor avaliação cruzada, passe o 'test_dataloader' padrão (com labels reais) na função!")
                x = batch[0]
                # Usa uma label fictícia apenas para o código não quebrar
                y = torch.zeros(x.size(0))
                
            x = x.to(device)
            z = F.normalize(model.embed(x), dim=1)
            embeddings.append(z.cpu())
            labels.append(y.cpu())
            
    embeddings = torch.cat(embeddings, dim=0) 
    labels = torch.cat(labels, dim=0) 
    
    if len(embeddings) > 2000:
        idx = torch.randperm(len(embeddings))[:2000]
        sub_emb = embeddings[idx]
        sub_labels = labels[idx]
    else:
        sub_emb = embeddings
        sub_labels = labels
    
    # Calcula a distância vetorial entre todo mundo (Matriz 2000 x 2000)
    dist_matrix = torch.cdist(sub_emb, sub_emb, p=2) 
    
    # Máscara para pegar distâncias de Classes Iguais (Positivos)
    same_class_mask = torch.eq(sub_labels.unsqueeze(1), sub_labels.unsqueeze(0))
    same_class_mask.fill_diagonal_(False) # Ignora a distância da imagem com ela mesma
    
    # Máscara para Classes Diferentes (Negativos)
    diff_class_mask = ~torch.eq(sub_labels.unsqueeze(1), sub_labels.unsqueeze(0))
    
    pos = dist_matrix[same_class_mask].numpy()
    neg = dist_matrix[diff_class_mask].numpy()

    # Plot das Curvas
    plt.figure()
    plt.hist(pos, bins=50, alpha=0.5, label="similar", density=True)
    plt.hist(neg, bins=50, alpha=0.5, label="different", density=True)

    # Threshold com Acurácia Balanceada (Sensibilidade e Especificidade)
    all_dist = np.concatenate([pos, neg])
    thresholds = np.linspace(all_dist.min(), all_dist.max(), 200)

    accs = []
    for t in thresholds: #find the accuracy of all thresholds
        tpr = np.sum(pos < t) / len(pos) if len(pos) > 0 else 0
        tnr = np.sum(neg > t) / len(neg) if len(neg) > 0 else 0
        acc = (tpr + tnr) / 2.0
        accs.append(acc)

    #find the best accuracy
    best_idx = np.argmax(accs)
    best_t = thresholds[best_idx]
    best_acc = accs[best_idx]

    plt.axvline(best_t, linestyle="--", label=f"thr={best_t:.2f}") #print the accuracy
    plt.legend()
    plt.title(title)
    # plt.show()

    print(f"Separation: {(neg.mean() - pos.mean()):.4f}")
    print(f"Balanced Accuracy: {best_acc:.4f}")

    plt.savefig(f"{file_path}/{title}.png")
    plt.close()

    return (neg.mean() - pos.mean()), best_acc

def get_embeddings(dataloader, model, device):
    model.eval()
    model.to(device)

    with torch.no_grad():
        embeddings = []
        labels = []

        for data, label in tqdm(dataloader):
            data = data.to(device)
            embedding = model.embed(data)
            embeddings.append(embedding.cpu().numpy())
            labels.append(label.cpu().numpy())

        return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)

def get_random_samples(data, labels, samples_per_class=200):
    """
    Amostra uma quantidade igual de exemplos para cada classe.
    Se a classe tiver menos amostras do que o solicitado, pega todas as disponíveis.
    """
    data = np.array(data)
    labels = np.array(labels)
    
    sampled_data = []
    sampled_labels = []
    
    # Unique classes
    unique_classes = np.unique(labels)
    
    for cls in unique_classes:
        cls_idx = np.where(labels == cls)[0]
        
        n_samples = min(samples_per_class, len(cls_idx))
        
        chosen_idx = np.random.choice(cls_idx, n_samples, replace=False)
        
        sampled_data.append(data[chosen_idx])
        sampled_labels.append(labels[chosen_idx])

    final_data = np.concatenate(sampled_data, axis=0)
    final_labels = np.concatenate(sampled_labels, axis=0)
    
    return final_data, final_labels

def plot_tsne(data, labels, title="t-SNE visualization", file_path="."):
    print("Running t-SNE...")

    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )
    
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
        
    data_2d = tsne.fit_transform(data)
    metrics_dict = {}
    
    # Calculate metrics
    metrics_dict = evaluate_original_embeddings(data_2d, labels)
    
    plt.figure(figsize=(8,8))

    scatter = plt.scatter(
        data_2d[:,0],
        data_2d[:,1],
        c=labels,
        cmap="tab10",
        s=10
    )

    plt.colorbar(scatter)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.grid(True)
    
    try:
        import os
        plt.savefig(os.path.join(file_path, f"{title}.png"))
    except Exception as e:
        print("Aviso: Falha ao salvar a imagem.", e)
        
    # plt.show()
    plt.close()

    return data_2d, metrics_dict


def evaluate_original_embeddings(data, labels):
    """
    Avalia a qualidade matemática dos embeddings em alta dimensão.
    """
    print("--- AVALIANDO EMBEDDINGS ---")
    
    # Previne erros caso sejam tensores do PyTorch
    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()

    # Redimensiona caso tenham mais de 2 dimensões (ex: batch_size, canais, largura, altura)
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    metrics = {}
    
    metrics["silhouette_score"] = float(silhouette_score(data, labels))
    metrics["davies_bouldin_index"] = float(davies_bouldin_score(data, labels))
    metrics["calinski_harabasz_index"] = float(calinski_harabasz_score(data, labels))
    
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.4f}")

    return metrics

