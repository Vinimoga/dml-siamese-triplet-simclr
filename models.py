import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import ContrastiveLoss, NTXentLoss

class MLP(nn.Module):
    """A simple Multi Layer perceptron for the
    purpose of testing very simple arquitetures"""
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 128,
            dropout=0.5,
    ):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.dropout = nn.Dropout(dropout)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = X.flatten(start_dim=1)
        #print(X.shape)
        X = F.relu(self.input_layer(X))
        #print(X.shape)
        X = self.dropout(X)
        X = F.relu(self.hidden_layer(X))
        #print(X.shape)
        X = self.dropout(X)
        X = self.output_layer(X)
        #print(X.shape)
        return X
    
    def __name__(self):
        return "MLP"
    
class LeNet(nn.Module):
    """A simple Convolutional Neural Network for the
    purpose of testing very simple arquitetures"""
    def __init__(self, input_dim, output_dim=1):
        super(LeNet, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)


        self.pool = nn.AvgPool2d(2)
        self.adaptive = nn.AdaptiveAvgPool2d((4,4)) #Added to use different datasets

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.embedding = nn.Linear(84, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x =  self.pool(x)
        x = F.relu(self.conv2(x))
        x =  self.pool(x)
        x = self.adaptive(x)
        x = torch.flatten(x, 1)
        #print(x.shape)  #[64, 256]
        x = F.relu(self.fc1(x))
        #print(x.shape)  #[64, 120]
        x = F.relu(self.fc2(x))
        #print(x.shape)  #[64, 84]
        x = self.embedding(x)
        #print(x.shape)  #[64, 1]
        return x
    
    def __name__(self):
        return "LeNet"
    
class AlexNet(nn.Module):
    """A big Convolutional Neural Network
       Adapted from a 256x256 colored image
       to a 28x28 or 32x32 from mnist and cifar"""
    def __init__(self, input_dim, output_dim=1):
        super(AlexNet, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.convs = nn.Sequential(
            nn.Conv2d(input_dim, 96, kernel_size=3, stride=1, padding=1),  # mudou de 11x11
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=output_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
    
    def __name__(self):
        return "AlexNet"
    

class BaseModel(nn.Module):
    def training_step(self, batch):
        raise NotImplementedError
    

class TripletNetwork(BaseModel):
    def __init__(self, backbone, criterion = nn.TripletMarginLoss(margin=1.0)):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion

    def forward(self, a, p, n):
        za = self.backbone(a)
        zp = self.backbone(p)
        zn = self.backbone(n)
        return za, zp, zn
    
    def embed(self, x):
        return self.backbone(x)

    def training_step(self, batch):
        a, p, n = batch
        za, zp, zn = self(a, p, n)
        loss = self.criterion(za, zp, zn)
        return loss
    
    def __name__(self):
        return f"TripletNetwork"
    
class SiameseNetwork(BaseModel):
    def __init__(self, backbone, criterion = ContrastiveLoss()):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        return z1, z2
    
    def embed(self, x):
        return self.backbone(x)

    def training_step(self, batch):
        x1, x2, label = batch
        z1, z2 = self(x1, x2)
        loss = self.criterion(z1, z2, label)
        return loss
    
    def __name__(self):
        return f"SiameseNetwork"

class SimCLR(BaseModel):
    #Based on https://github.com/Spijkervet/SimCLR and https://arxiv.org/pdf/2002.05709.pdf
    def __init__(self, backbone, projection_dim=64, temperature=0.5):
        super().__init__()

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(backbone.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )

        self.criterion = NTXentLoss(temperature)

    def embed(self, x):
        return self.backbone(x)

    def project(self, z):
        return self.projector(z)

    def forward(self, x1, x2):
        h1 = self.embed(x1)
        h2 = self.embed(x2)

        z1 = self.project(h1)
        z2 = self.project(h2)

        return z1, z2

    def training_step(self, batch):
        x1, x2, _ = batch # label is not used in SimCLR (self-supervised)

        z1, z2 = self(x1, x2)
        loss = self.criterion(z1, z2)

        return loss

    def __name__(self):
        return "SimCLR"