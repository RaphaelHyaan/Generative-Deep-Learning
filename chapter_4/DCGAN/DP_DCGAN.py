import torch
from torch import nn, optim
from torch.autograd.variable import Variable
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DCGAN.DP_discriminator import Discriminator
from DCGAN.DP_generator import Generator

class DCGAN(nn.Module):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def forward(self, real_images):
        batch_size = real_images.shape[0]
        random_latent_vectors = torch.randn((1,batch_size, self.latent_dim)).cuda()

        generated_images = self.generator(random_latent_vectors)
        real_predictions = self.discriminator(real_images)
        fake_predictions = self.discriminator(generated_images)

        return real_predictions, fake_predictions

import matplotlib.pyplot as plt

def train(dcgan, data_loader, d_optimizer, g_optimizer, loss_fn, epochs):
    # 创建一个列表来保存生成的图像
    generated_images = []

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(data_loader):
            d_optimizer.zero_grad()

            real_images = real_images.cuda()

            real_predictions, fake_predictions = dcgan(real_images)

            real_labels = torch.ones_like(real_predictions)
            real_noisy_lables = real_labels - 0.1 * torch.rand_like(real_predictions)
            fake_labels = torch.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels + 0.1 * torch.rand_like(fake_predictions)

            d_loss_real = loss_fn(real_predictions, real_noisy_lables)
            d_loss_fake = loss_fn(fake_predictions, fake_noisy_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()

            g_real_predictions, g_fake_predictions = dcgan(real_images)
            g_loss = loss_fn(g_fake_predictions, real_labels)

            
            g_loss.backward()
            g_optimizer.step()

            if i % 50 == 0:
                print(f"Epoch {epoch}, Batch {i}, D loss: {d_loss.item()}, G loss: {g_loss.item()}")

        # 每5个epoch生成并保存10张图像
        if epoch % 5 == 0:
            with torch.no_grad():
                fake_images = dcgan.generator(torch.randn(10, 100).cuda())
                generated_images.append(fake_images)

    # 训练结束后显示生成的图像
    for i, images in enumerate(generated_images):
        plt.figure(figsize=(10,10))
        for j, image in enumerate(images):
            plt.subplot(10, 10, j+1)
            plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0)* 127.5 + 127.5)
            plt.axis('off')
        plt.show()
