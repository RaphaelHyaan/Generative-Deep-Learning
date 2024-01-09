import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

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

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,stride = 2,padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride = 2,padding = 1),
            nn.ReLU(),
            nn.Conv2d(64,128,3,stride = 2,padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*4*4,2)
        )
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
        x = self.decoder(x)
        return x

model = Autoencoder().cuda()
optimizer = optim.Adam(model.parameters())
loss_function = nn.BCELoss().cuda()

for epoch in range(20):
    for data, _ in trainloader:
        data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, data)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    for data, _ in testloader:
        data = data.cuda()
        output = model(data)
        loss = nn.BCELoss()(output, data)
    print(f'Test Loss: {loss.item()}')

