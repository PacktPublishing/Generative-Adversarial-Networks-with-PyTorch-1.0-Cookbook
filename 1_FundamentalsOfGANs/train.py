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

from generator import Generator, z_sampler
from discriminator import Discriminator
from utils import create_session_dir

def get_loaders(cfg):
    train_loader = DataLoader(
            tv.datasets.MNIST("data", train=True, download=True,
                transform=tv.transforms.ToTensor()),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"])
    return train_loader

def save_sample_images(m_gen, epoch, cfg):
    z = z_sampler(cfg["batch_size"], cfg["z_dim"], cfg["cuda"])
    xhat = m_gen(z)
    xhat = F.interpolate(xhat, scale_factor=(5.0, 5.0))
    tv.utils.save_image(xhat, "samples/%03d.png" % epoch)

def train(m_gen, m_disc, train_loader, optimizers, cfg):
    cudev = cfg["cuda"]
    batch_size = cfg["batch_size"]
    optD,optG = optimizers
    d_criterion = lambda yhat,y : -torch.mean( y*torch.log(yhat) \
            + (1-y)*torch.log(1-yhat) )
    g_criterion = lambda yhat : -torch.mean( torch.log(yhat) )
    for epoch in range( cfg["num_epochs"] ):
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
            d_fake_loss = d_criterion(m_disc(fake_x), fake_labels)
            d_real_loss = d_criterion(m_disc(real_x), real_labels)
            d_loss = d_fake_loss + d_real_loss

            d_loss.backward()
            optD.step()

            optG.zero_grad()
            fake_x2 = m_gen( z_sampler(batch_size, cfg["z_dim"], cudev) )
            g_loss = g_criterion(m_disc(fake_x2))
            g_loss.backward()
            optG.step()

            print("GLoss: %.4f, DLossReal: %.4f, DLossFake: %.4f, " \
                    % (g_loss.item(), d_real_loss.item(), d_fake_loss.item()))
            
        save_sample_images(m_gen, epoch, cfg)

def main(args):
    cfg = vars(args)
    cfg["session_dir"] = create_session_dir()
    init_session_log(cfg)
    train_loader = get_loaders(cfg)
    m_gen = Generator(cfg["z_dim"])
    m_disc = Discriminator()
    cudev = cfg["cuda"]
    if cudev >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA device specified but CUDA not available")
    if cudev >= 0:
        m_gen.cuda(cudev)
        m_disc.cuda(cudev)
    optG = torch.optim.SGD([{"params" : m_gen.parameters()}], lr=cfg["lr_g"],
        momentum=cfg["momentum"])
    optD = torch.optim.SGD([{"params" : m_disc.parameters()}], lr=cfg["lr_d"],
        momentum=cfg["momentum"])
    train(m_gen, m_disc, train_loader, (optD,optG), cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=-1,
            help="Cuda device number, select -1 for cpu")
    parser.add_argument("--num-workers", type=int, default=4,
        help="Number of worker threads to use loading data")
    parser.add_argument("--num-epochs", type=int, default=100)
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
