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
        self._fc1 = nn.Linear(z_dim, 150, bias=False)
        self._fc2 = nn.Linear(150, 784, bias=False)

        nn.init.normal_(self._fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self._fc2.weight, mean=0, std=0.1)

    def forward(self, x):
        x = F.relu( self._fc1(x) )
        x = torch.sigmoid( self._fc2(x) )
        x = x.view(-1, 1, 28, 28)
        return x

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

def save_sample_images(m_gen, epoch, cfg):
    z = z_sampler(cfg["batch_size"], cfg["z_dim"], cfg["cuda"])
    xhat = m_gen(z)
    xhat = F.interpolate(xhat, scale_factor=(5.0, 5.0))
    tv.utils.save_image(xhat, "samples/%03d.png" % epoch)

def train(m_gen, m_disc, train_loader, test_loader, optimizers, cfg):
    cudev = cfg["cuda"]
    batch_size = cfg["batch_size"]
    optD,optG = optimizers
    d_criterion = lambda yhat,y : -torch.mean( y*torch.log(yhat) \
            + (1-y)*torch.log(1-yhat) )
    g_criterion = lambda yhat : -torch.mean( torch.log(yhat) )
    criterion = nn.BCELoss()
    for epoch in range(100):
        for i,(real_x,_) in enumerate(train_loader):
            real_labels = torch.ones(batch_size).cuda(cudev)
            fake_labels = torch.zeros(batch_size).cuda(cudev)
            if cudev >= 0:
                real_x = real_x.cuda(cudev)
                real_labels = real_labels.cuda(cudev)
                fake_labels = fake_labels.cuda(cudev)
                labels = torch.cat((torch.ones(batch_size), 
                    torch.zeros(batch_size))).cuda(cudev)
            z = z_sampler(batch_size, cfg["z_dim"], cudev)

            optD.zero_grad()
            fake_x = m_gen(z)
            d_fake_loss = criterion(m_disc(fake_x), fake_labels)
            d_real_loss = criterion(m_disc(real_x), real_labels)
            d_loss = d_fake_loss + d_real_loss

            d_loss.backward()
            optD.step()

            optG.zero_grad()
            fake_x2 = m_gen( z_sampler(batch_size, cfg["z_dim"], cudev) )
            g_loss = criterion(m_disc(fake_x2), real_labels)
            g_loss.backward()
            optG.step()

            print("GLoss: %.4f, DLossReal: %.4f, DLossFake: %.4f, " \
                    % (g_loss.item(), d_real_loss.item(), d_fake_loss.item()))
            
        save_sample_images(m_gen, epoch, cfg)

def z_sampler(batch_size, z_dim, cudev):
    if cudev >= 0:
        z = torch.cuda.FloatTensor(batch_size, z_dim).normal_(0.0,1.0)
    else:
        z = torch.FloatTensor(batch_size, z_dim).normal_(0.0,1.0)
    return z

def main(args):
    cfg = vars(args)
    train_loader,test_loader = get_loaders(cfg)
    m_gen = Generator(cfg["z_dim"])
    m_disc = Discriminator()
    cudev = cfg["cuda"]
    if cudev >= 0:
        m_gen.cuda(cudev)
        m_disc.cuda(cudev)
    optG = torch.optim.SGD([{"params" : m_gen.parameters()}], lr=cfg["lr_g"],
        momentum=cfg["momentum"])
    optD = torch.optim.SGD([{"params" : m_disc.parameters()}], lr=cfg["lr_d"],
        momentum=cfg["momentum"])
    train(m_gen, m_disc, train_loader, test_loader, (optD,optG), cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0,
            help="Cuda device number, select -1 for cpu")
    parser.add_argument("--num-workers", type=int, default=4,
        help="Number of worker threads to use loading data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=100,
        help="Number of latent space units")
    parser.add_argument("--lr-d", type=float, default=0.0001,
            help="Model learning rate")
    parser.add_argument("--lr-g", type=float, default=0.001,
            help="Model learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
            help="Momentum parameter for the SGD optimizer")
    args = parser.parse_args()
    main(args)
