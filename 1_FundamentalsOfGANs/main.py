"""
This is a very simple, self-contained GAN which trains on MNIST and generates
samples accordingly.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.utils.funcional as F

class Generator(nn.Module):
    def __init__(self, z_dim=10):
        self._fc1 = nn.Linear(z_dim, 20)
        self._fc2 = nn.Linear(20, 150)
        self._fc3 = nn.Linear(150, 784)

    def forward(self, x):
        x = F.relu( self._fc1(x) )
        x = F.relu( self._fc2(x) )
        x = F.sigmoid( self._fc3(x) )
        x = x.view(-1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        self._fc1 = nn.Linear(784, 150)
        self._fc2 = nn.Linear(150, 20)
        self._fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu( self._fc1(x) )
        x = F.relu( self._fc2(x) )
        x = F.sigmoid( self._fc3(x) )
        return x


