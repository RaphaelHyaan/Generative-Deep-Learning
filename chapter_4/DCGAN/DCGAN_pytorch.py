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
from DCGAN.DP_discriminator import Discriminator
from DCGAN.DP_generator import Generator
from DCGAN.DP_DCGAN import DCGAN,train
from utiles import display

CUDA_LAUNCH_BLOCKING=1

device =  torch.device('cuda')

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255.0 - 127.5) / 127.5)
])

train_data = datasets.ImageFolder(root = "chapter_4/dataset_dp",transform=transform)
train_loader = DataLoader(train_data,batch_size=128,shuffle=True)

# Plot some training images
real_batch = next(iter(train_loader))

# Convert the batch to numpy array
real_batch_images = real_batch[0]

real_batch_images = torch.squeeze(real_batch_images).numpy()

# Display the images
# display(real_batch_images, cmap="gray_r", as_type="float32")

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
loss_fn = nn.BCELoss()

train(dcgan, train_loader, d_optimizer, g_optimizer, loss_fn, 100)
# 在测试阶段，我们通常不需要计算梯度
with torch.no_grad():
    # 创建一个随机潜在向量
    latent_vector = torch.randn(64, 100).to(device)
    # 使用生成器生成图像
    generated_images = dcgan.generator(latent_vector)
    # 将生成的图像从[-1, 1]的范围转换到[0, 1]的范围
    generated_images = (generated_images + 1) / 2
    # 将生成的图像从GPU移动到CPU，并转换为numpy数组
    generated_images = torch.squeeze(generated_images).cpu().numpy()
    # 显示生成的图像
    display(generated_images)