
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

import numpy as np

from model import *

df_training = torchvision.datasets.DatasetFolder('../my_things/llsdata2/',
                                        loader = np.load, extensions='npz',
                                                 transform=transform_train)
dataloader = torch.utils.data.DataLoader(df_training,
                                           batch_size=batch_size, shuffle=True)


model = torch.load('conv_autoencoder3.pth')
model.eval()

import matplotlib.pyplot as plt

x = []
for i in range(20):
    for data in dataloader:
        print(i)
        img, _ = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        else:
            img = Variable(img)
        # ===================forward=====================
        output = model.latent(img)
        x.append(output.cpu().detach().numpy().reshape([-1, output.shape[1]]))

x = np.vstack(x)
plt.hist2d(x[:,0], x[:,2], 100)

plt.plot(x[:,0], x[:,1], '.')


### Plot Latent space cutouts

X, Y = np.meshgrid(np.linspace(-3,3,32), np.linspace(-3,3,32))
Z = np.zeros([len(X.flatten()), 1])
tt = np.vstack([X.flatten(), Y.flatten(), Z.T]).T
tt = torch.tensor(tt, device='cuda', dtype= torch.float32).roll(2,0)
res = model.decoder(tt)
res = res.cpu().detach().numpy()

map = np.zeros([32*8, 32*8])
for i in range(32):
    for j in range(32):
        t = res[i*32+j, 0, :, :]
        map[(i*8):(i*8+8), (j*8):(j*8+8)] = t # / t.std()

plt.imshow(map)

