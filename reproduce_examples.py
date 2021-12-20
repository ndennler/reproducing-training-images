import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision.utils as vutils

import itertools
import numpy as np
import matplotlib.pyplot as plt
import copy

# class DataReproducer():
#     def __init__(self):
device='cpu'#cuda:0'
device = 'cuda:0'


print('loading models...')
models = []
for name in [3,4,5]:
  model = torch.load(f'/home/nathan/Desktop/reproducing-training-images/trained_models/Autoencoder{name}.pth')
  models.append(model)

# model3 = torch.load('/home/nathan/Desktop/reproducing-training-images/trained_models/Autoencoder3.pth')
# models=[model1, model2]
print('models loaded.')

optimized_input = torch.randn((8,3,64,64), device=device, dtype=torch.float32, requires_grad=True)
# pre_optimized_input = copy.deepcopy(optimized_input)

input_optimizer = torch.optim.Adam([optimized_input], lr=.2) 

models = [model.eval().to(device) for model in models]

activation= nn.Sigmoid()
activation= nn.Tanh()

criterion = nn.MSELoss() 

for i in range(34):
    if i % 10 == 0:
        ims = optimized_input.data.cpu().numpy()
        # im = optimized_input[3].data.cpu().numpy()
        # im = (im - np.min(im)) / np.ptp(im)
        # ims.append(im)

        for model in models:
          ims = np.vstack((ims, model.reconstruct(optimized_input).data.cpu().numpy()))

        fig = plt.figure(figsize = (20, 5))   
        ax1 = plt.subplot(111)

        plt.imshow(np.transpose(vutils.make_grid(torch.tensor(ims), padding=2, normalize=True, nrow=optimized_input.shape[0], value_range=(-1,1)).cpu(),(1,2,0)))
        plt.title(f'Iteration {i}')
        plt.show()

    input_optimizer.zero_grad()
    
    loss = 0

    for model1, model2 in itertools.combinations(models,2):
    # for model1 in models:
    #   for model2 in models:
        output1 = model1.reconstruct(optimized_input)
        output2 = model2.reconstruct(optimized_input)

        loss += criterion(output1,output2)

    loss.backward()
    input_optimizer.step()
    print(loss.item())