import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.convt1 = nn.ConvTranspose2d(100,512,kernel_size=4,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(512,momentum=0.9)
        self.convt2 = nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(256,momentum=0.9)
        self.convt3 = nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(128,momentum=0.9)
        self.convt4 = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(64,momentum=0.9)
        self.convt5 = nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=1,bias=False)
    
    def forward(self,x):
        x = x.view(-1,100,1,1)
        x = F.relu(self.bn1(self.convt1(x)))
        x = F.relu(self.bn2(self.convt2(x)))
        x = F.relu(self.bn3(self.convt3(x)))
        x = F.relu(self.bn4(self.convt4(x)))
        x = torch.tanh(self.convt5(x))
        return x

if __name__ == '__main__':
    # Create an instance of the Generator class
    generator = Generator()

    # Generate a random input tensor
    input_tensor = torch.randn(100, )

    # Pass the input tensor through the generator
    output_tensor = generator(input_tensor)

    # Print the shape of the output tensor
    print(output_tensor.shape)

        