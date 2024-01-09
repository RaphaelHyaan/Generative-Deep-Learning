import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_SIZE = 32
BATCH_SIZE = 100
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 5
BETA = 500

device =  torch.device('cuda')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Pad(2)
])

trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class Sampling(nn.Module):
    def forward(self,inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(batch, dim).cuda()
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,stride = 2,padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride = 2,padding = 1),
            nn.ReLU(),
            nn.Conv2d(64,128,3,stride = 2,padding = 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.ZMEAN = nn.Linear(128*4*4,EMBEDDING_DIM)
        self.ZLOGVAR = nn.Linear(128*4*4,EMBEDDING_DIM)
        self.Sampling = Sampling()
        self.decoder = nn.Sequential(
            nn.Linear(2,128*4*4),
            nn.Unflatten(1,(128,4,4)),
            nn.ConvTranspose2d(128,64,3,stride = 2,padding = 1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()            
        )
    def forward(self,x):
        x = self.encoder(x)
        z_mean = self.ZMEAN(x)
        z_log_var = self.ZLOGVAR(x)
        x = self.Sampling([z_mean,z_log_var])
        x = self.decoder(x)
        return z_mean,z_log_var,x
    def loss_function(self,x,z_mean,z_log_var,reconstruction):
        # 重构损失
        reconstruction_loss = F.binary_cross_entropy(reconstruction,x,reduction = 'sum')
        # kl散度
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(),axis=1))
        return reconstruction_loss + kl_loss

vae = VAE().cuda()
optimizer = torch.optim.Adam(vae.parameters(),lr = 0.0005)

for epoch in range(EPOCHS):
    for x,_ in trainloader:
        x = x.cuda()
        z_mean,z_log_var,reconstruction = vae(x)
        loss = vae.loss_function(x,z_mean,z_log_var,reconstruction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_loss = 0
    for y,_ in testloader:
        y = y.cuda()
        z_mean,z_log_var,reconstruction = vae(y)
        test_loss += vae.loss_function(y,z_mean,z_log_var,reconstruction)
    test_loss /= len(testloader)
    print(f"Epoch: {epoch+1},  Test Loss: {test_loss}")

print(vae)