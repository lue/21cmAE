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
from transforms import *

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.cov1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.ReLU1 = nn.ReLU(True)
        self.MaxPool2d1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.cov2 = nn.Conv2d(16, 5, 3, stride=1, padding=1)
        self.ReLU2 = nn.ReLU(True)
        self.cov3 = nn.Conv2d(5, 2, 3, stride=1, padding=1)
        self.MaxPool2d2 = nn.MaxPool2d(2, stride=1, return_indices=True)
        self.lin1 = nn.Linear(32, 3)

        self.b_lin1 = nn.Linear(3, 32)
        self.b_cov1 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        self.b_ReLU1 = nn.ReLU(True)
        self.b_cov2 = nn.ConvTranspose2d(5, 16, 3, stride=2, padding=1, output_padding=1)
        self.b_ReLU2 = nn.ReLU(True)
        self.b_cov3 = nn.ConvTranspose2d(2, 5, 3, stride=1, padding=1, output_padding=0)
        self.b_tanh = nn.Tanh()

    def encoder(self, x):
        x = self.cov1(x)
        print(x.shape)
        x = self.ReLU1(x)
        print(x.shape)
        # x = self.MaxPool2d1(x)
        x = self.cov2(x)
        print(x.shape)
        x = self.ReLU2(x)
        print(x.shape)
        x = self.cov3(x)
        print(x.shape)
        # x = self.MaxPool2d2(x)
        x = x.view(-1, 32)
        x = self.lin1(x)
        print(x.shape)
        return x

    def decoder(self, x):
        x = self.b_lin1(x)
        print('dec: ', x.shape)
        x = x.view(-1,2,4,4)
        print('dec: ', x.shape)
        x = self.b_cov3(x)
        print('dec: ', x.shape)
        x = self.b_cov2(x)
        x = self.b_ReLU1(x)
        # print('dec: ', x.shape)
        # print(x.shape)
        x = self.b_cov1(x)
        # print('dec: ', x.shape)
        # x = self.b_ReLU2(x)
        # x = self.b_cov3(x)
        x = self.b_tanh(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x

    def latent(self, x):
        x = self.encoder(x)
        return x

