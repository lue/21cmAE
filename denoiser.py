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
import matplotlib.pyplot as plt


df_training = torchvision.datasets.DatasetFolder('../my_things/llsdata2/',
                                        loader = np.load, extensions='npz',
                                                 transform=transform_train)
dataloader = torch.utils.data.DataLoader(df_training,
                                           batch_size=batch_size, shuffle=True)


# model = torch.load('conv_autoencoder2.pth')
# model.eval()

model = torch.load('conv_autoencoder3.pth')
model.eval()

map = np.load('../my_things/llsdata2/train/data_100.npy.npz')['arr_0']
map = np.load('../eor_data/train/data_1_0.npy.npz')['arr_0']

N = int(map.shape[0] / 2)

map = np.pad(map, 3, mode='wrap')
ax = plt.subplot(231)
plt.imshow(map[3:-3:2,3:-3:2])

map += np.random.normal(0, 100.8, size=map.shape)



plt.subplot(232,sharex=ax, sharey=ax)
plt.imshow(map[3:-3:2,3:-3:2])

res = np.zeros([N*N, 1, 8, 8])
temp = RandomLogNorm(0)
for i in range(N):
    for j in range(N):
        res[i*N+j, 0, :, :] = map[(i*2):(i*2+8), (j*2):(j*2+8)]

res_m = res.mean((2,3))
res_s = res.std((2,3))

for i in range(N):
    for j in range(N):
        res[i * N + j, 0, :, :] = temp(res[i * N + j, 0, :, :] )

res = torch.tensor(res, device='cuda', dtype= torch.float32)
res2 = model.encoder(res)
res = model.decoder(res2).cpu().detach().numpy()
res2 = res2.cpu().detach().numpy()


plt.subplot(234,sharex=ax, sharey=ax)
plt.imshow(res2[:,0].reshape([N,N]))
plt.subplot(235,sharex=ax, sharey=ax)
plt.imshow(res2[:,1].reshape([N,N]))
plt.subplot(236,sharex=ax, sharey=ax)
plt.imshow(res2[:,2].reshape([N,N]))

for i in range(res.shape[0]):
    res[i,:,:,:] *= res_s[i]*2
    res[i,:,:,:] += res_m[i]

map2 = np.zeros([2*N+6, 2*N+6])
for i in range(N):
    for j in range(N):
        map2[(i*2+3):(i*2+5), (j*2+3):(j*2+5)] = res[i*N+j, 0, 3:5, 3:5]

plt.subplot(233,sharex=ax, sharey=ax)
plt.imshow(map2[3:-3:2,3:-3:2])

# plt.hist([res2[:,0], res3[:,0]], bins=1000, normed=True, cumulative=True, histtype='step')

# plt.hist([resn[:,0], res2[:,0], res3[:,0]], bins=1000, normed=True, cumulative=True, histtype='step')