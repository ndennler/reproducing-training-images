from dataloaders.celebA import CelebAImageDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from model_specifications.celebA_VAE import AutoEncoder

from PIL import Image
import matplotlib.pyplot as plt


# Set random seed for reproducibility
manualSeed = 69
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# ----------------------------------   LOADING DATA   --------------------------
image_size = 512
batch_size = 32
numworkers = 4

dataset = CelebAImageDataset('/home/nathan/Desktop/reproducing-training-images/data/celebA/identity_CelebA.txt', 
                '/home/nathan/Desktop/reproducing-training-images/data/celebA/img_align_celeba', 
                transform=transforms.Compose([
                               Image.fromarray,
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
            )

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=numworkers)


# ----------------------------------   SETTING UP MODEL   ----------------------

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

nz = 256
device = 'cuda:0'
learning_rate=5e-3


model = AutoEncoder(nz)
model.apply(weights_init)

model.to(device)

opt = torch.optim.Adam(model.parameters(), learning_rate)
train_it = 0
epochs = 4

# Plot some training images
real_batch = next(iter(dataloader))
print(real_batch.shape)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

out = model(real_batch.to(device))['rec']
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Reconstructed Images")
plt.imshow(np.transpose(vutils.make_grid(out[:64], padding=2, normalize=True).cpu(),(1,2,0)))  
plt.show()

model.train()

for _ in range(epochs):
    for batch in tqdm(dataloader):
        model.zero_grad()
        sample_img_gpu = batch.to(device)

        out = model.forward(sample_img_gpu)

        total_loss, losses = model.loss(sample_img_gpu, out)
        total_loss.backward()
        opt.step()

        if train_it % 100 == 0:
            # print("It {}: Reconstruction Loss: {}".format(train_it, total_loss))
            print("It {}: Reconstruction Loss: {}; rec: {}, KL: {}".format(train_it, total_loss, losses['rec_loss'], losses['kl_loss']))

        train_it += 1
        

torch.save(model, 'trained_models/Autoencoder_512_2.pth') 

# Plot some training images
real_batch = next(iter(dataloader))
print(real_batch.shape)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

model.eval()
out = model(real_batch.to(device))['rec']
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Reconstructed Images")
plt.imshow(np.transpose(vutils.make_grid(out[:64], padding=2, normalize=True).cpu(),(1,2,0)))  
plt.show()
