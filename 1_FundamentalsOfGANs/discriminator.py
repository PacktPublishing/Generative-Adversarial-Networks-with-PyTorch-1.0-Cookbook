"""
The Generator for our simple MNIST GAN.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self._fc1 = nn.Linear(784, 150, bias=False)
        self._fc2 = nn.Linear(150, 1, bias=False)

        nn.init.normal_(self._fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self._fc2.weight, mean=0, std=0.1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu( self._fc1(x) )
        x = torch.sigmoid( self._fc2(x) )
        return x


def _test_main(args):
    discriminator = Discriminator()
    if args.cuda >= 0:
        x = torch.cuda.FloatTensor(args.batch_size, 1, 28, 28).uniform_(0,1)
    else:
        x = torch.FloatTensor(args.batch_size, 1, 28, 28).uniform_(0,1)
    pred = discriminator(x)
    print("Predictions:")
    print(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=-1,
            help="Cuda device number, select -1 for cpu")
    args = parser.parse_args()
    _test_main(args)

