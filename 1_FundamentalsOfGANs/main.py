"""
This is a very simple, self-contained GAN which trains on MNIST and generates
samples accordingly.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, z_dim=10):
        super().__init__()
        self._fc1 = nn.Linear(z_dim, 20)
        self._fc2 = nn.Linear(20, 150)
        self._fc3 = nn.Linear(150, 784)

    def forward(self, x):
        x = F.relu( self._fc1(x) )
        x = F.relu( self._fc2(x) )
        x = F.sigmoid( self._fc3(x) )
        x = x.view(-1, 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self._fc1 = nn.Linear(784, 150)
        self._fc2 = nn.Linear(150, 20)
        self._fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu( self._fc1(x) )
        x = F.relu( self._fc2(x) )
        x = F.sigmoid( self._fc3(x) )
        return x


def get_loaders(cfg):
    train_loader = DataLoader(
            tv.datasets.MNIST("data", train=True, download=True,
                transform=tv.transforms.ToTensor()),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"])
    test_loader = DataLoader(
            tv.datasets.MNIST("data", train=False,
                transform=tv.transforms.ToTensor()),
            batch_size=cfg["batch_size"],
            shuffle=True, 
            num_workers=cfg["num_workers"])
    return train_loader,test_loader

def train(m_gen, m_disc, train_loader, test_loader, optimizer, cfg):
    cudev = cfg["cuda"]
    batch_size = cfg["batch_size"]
    d_criterion = lambda yhat,y : torch.mean( y*torch.log(yhat) \
            + (1-y)*torch.log(1-yhat) )
    g_criterion = lambda yhat,y : torch.mean( torch.log(yhat) )
    for epoch in range(100):
        for x,_ in train_loader:
            if cudev >= 0:
                z = torch.cuda.FloatTensor(cfg["z_dim"]).uniform_(0.0,1.0)
                x = x.cuda(cudev)
                labels = torch.cat((torch.ones(batch_size), 
                    torch.zeros(batch_size))).cuda(cudev)
            else:
                z = torch.FloatTensor(cfg["z_dim"]).uniform_(0.0,1.0)
                labels = torch.cat((torch.ones(batch_size), 
                    torch.zeros(batch_size)))
            
            optimizer.zero_grad()
            xhat = m_gen(z)
            x = torch.cat((x, xhat))
            preds = m_disc(x)
            d_loss = d_criterion(preds, labels)
            g_loss = g_criterion(preds[batch_size:], labels[batch_size:])
            print(d_loss.item(), g_loss.item())
            d_loss.backward(retain_graph=True)
            g_loss.backward()
            optimizer.step()


def main(args):
    cfg = vars(args)
    train_loader,test_loader = get_loaders(cfg)
    m_gen = Generator(cfg["z_dim"])
    m_disc = Discriminator()
    cudev = cfg["cuda"]
    if cudev >= 0:
        m_gen.cuda(cudev)
        m_disc.cuda(cudev)
    optimizer = torch.optim.SGD([{"params" : m_gen.parameters()},
        {"params" : m_disc.parameters()}], lr=cfg["lr"],
        momentum=cfg["momentum"])
    train(m_gen, m_disc, train_loader, test_loader, optimizer, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0,
            help="Cuda device number, select -1 for cpu")
    parser.add_argument("--num-workers", type=int, default=4,
        help="Number of worker threads to use loading data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=10,
        help="Number of latent space units")
    parser.add_argument("--lr", type=float, default=0.0001,
            help="Model learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
            help="Momentum parameter for the SGD optimizer")
    args = parser.parse_args()
    main(args)
