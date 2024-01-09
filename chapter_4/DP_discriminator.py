import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv2 = nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(128,momentum=0.9)
        self.conv3 = nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(256,momentum=0.9)
        self.conv4 = nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(512,momentum=0.9)
        self.conv5 = nn.Conv2d(512,1,kernel_size=4,stride=1,padding=0,bias=False)

    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.dropout(x,0.3)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.conv5(x))
        return x.flatten()


if __name__ == '__main__':
    # Create an instance of the Discriminator model
    model = Discriminator().cuda()

    # Generate a random input tensor
    batch_size = 64
    input_tensor = torch.randn(batch_size, 1, 64, 64).cuda()

    # Pass the input tensor through the model
    output = model(input_tensor)

    # Print the output tensor
    print(output)
