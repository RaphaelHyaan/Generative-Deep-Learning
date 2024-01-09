import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_4.DP_discriminator import Discriminator
from chapter_4.DP_generator import Generator
from chapter_4.DP_DCGAN import DCGAN,train

device =  torch.device('cuda')

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_data = datasets.ImageFolder(root = "chapter_4/dataset_dp",transform=transform)
train_loader = DataLoader(train_data,batch_size=128,shuffle=True)
# Create the discriminator and generator
discriminator = Discriminator()
generator = Generator()

# Move models to GPU
discriminator.cuda()
generator.cuda()

# Create the DCGAN

dcgan = DCGAN(discriminator, generator, 100).cuda()
d_optimizer = optim.Adam(dcgan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(dcgan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss_fn = nn.BCEWithLogitsLoss()

train(dcgan, train_loader, d_optimizer, g_optimizer, loss_fn, 5)
# 在测试阶段，我们通常不需要计算梯度
with torch.no_grad():
    # 创建一个随机潜在向量
    latent_vector = torch.randn(64, 100).to(device)
    # 使用生成器生成图像
    generated_images = dcgan.generator(latent_vector)
    # 将生成的图像从[-1, 1]的范围转换到[0, 1]的范围
    generated_images = (generated_images + 1) / 2
    # 将生成的图像从GPU移动到CPU，并转换为numpy数组
    generated_images = generated_images.cpu().numpy()
    # 显示生成的图像
    plt.figure(figsize=(10, 10))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.show()