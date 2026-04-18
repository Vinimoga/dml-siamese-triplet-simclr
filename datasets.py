
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

        # índice por classe
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(base_dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_anchor, label_anchor = self.dataset[index]

        # POSITIVE (mesma classe)
        index_pos = index
        while index_pos == index:
            index_pos = random.choice(self.class_to_indices[label_anchor])

        img_positive, _ = self.dataset[index_pos]

        # NEGATIVE (classe diferente)
        label_neg = label_anchor
        while label_neg == label_anchor:
            label_neg = random.choice(self.classes)

        index_neg = random.choice(self.class_to_indices[label_neg])
        img_negative, _ = self.dataset[index_neg]

        return img_anchor, img_positive, img_negative
    
class SiameseDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

        # índice de classes
        #self.labels = []
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(base_dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        img1, label1 = self.dataset[index]

        # 50% positivo / 50% negativo
        if random.random() < 0.5:

            # par positivo
            index2 = index
            while index2 == index:
                index2 = random.choice(self.class_to_indices[label1])

            img2, _ = self.dataset[index2]
            label = 0  # similar

        else:

            # par negativo
            label2 = label1
            while label2 == label1:
                label2 = random.choice(self.classes)

            index2 = random.choice(self.class_to_indices[label2])
            img2, _ = self.dataset[index2]
            label = 1  # diferente

        return img1, img2, torch.tensor(label, dtype=torch.float32)

class SimCLRDataset(Dataset):
    #Based on https://github.com/Spijkervet/SimCLR and https://arxiv.org/pdf/2002.05709.pdf
    def __init__(self, base_dataset, transform=None):
        self.dataset = base_dataset
        if transform is None:
            self.transform = self.get_simclr_transform()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]

        x1 = self.transform(x)
        x2 = self.transform(x)

        return x1, x2
    
    def get_simclr_transform(self, size=32):
        return T.Compose([
            T.RandomResizedCrop(size),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor(),
        ])