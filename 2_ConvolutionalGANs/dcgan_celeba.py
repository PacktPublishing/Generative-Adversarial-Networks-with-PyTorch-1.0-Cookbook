"""
This is a DCGAN following Radford et al. 2015 which will train on Celeb-A and
generate images accordingly.
"""

import argparse
import logging
import os
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

from make_conv_layers import ConvLayers, DeconvLayers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_session_dir, init_session_log

pe = os.path.exists
pj = os.path.join

class Generator(DeconvLayers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = x.view((-1, x.shape[1], 1, 1))
        x = super().forward(x)
        x = torch.tanh(x)
        return x

class Discriminator(ConvLayers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fc = None
        self._last_conv = None

        c_out = self._num_base_chans*( 2**(self._num_layers-1) )
        self._last_conv = nn.Conv2d(c_out, 1, kernel_size=self._kernel_size,
                stride=1, padding=0)

    def forward(self, x):
        x = super().forward(x)
        x = self._last_conv(x)
        x = torch.sigmoid(x)
        return x

class ImageFolder(Dataset):
    def __init__(self, data_path, transform=None, ext=".png"):
        self._images = [pj(data_path,f) for f in os.listdir(data_path) \
                if f.endswith(ext)]
        self._transform = tv.transforms.ToTensor() if transform == None \
                else transform

    def __getitem__(self, index):
        return self._transform( Image.open(self._images[index]) ), -1

    def __len__(self):
        return len(self._images)


def get_loader(cfg):
    input_size = 2**(cfg["num_layers"] + 1)
    train_transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(brightness=0.25, contrast=0.25,
            saturation=0.25, hue=0.025),
        tv.transforms.Resize(input_size),
        tv.transforms.ToTensor()
        ])
    celeba_dataset = ImageFolder(cfg["celeba_path"], transform=train_transform,
            ext=".jpg")
    train_loader = DataLoader(dataset=celeba_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"])
    return train_loader

def make_first_batch(train_loader, cfg):
    fb_dir = pj(cfg["session_dir"], "first_batch")
    os.makedirs(fb_dir)
    cudev = cfg["cuda"]
    for real_x,_ in train_loader:
        if cudev >= 0:
            real_x = real_x.cuda(cudev)
        break
    tv.utils.save_image(real_x, pj(fb_dir, "first_batch.png"))

def save_sample_images(m_gen, epoch, cfg):
    z = z_sampler(cfg["batch_size"], cfg["z_dim"], cfg["cuda"])
    xhat = m_gen(z)
    xhat = F.interpolate(xhat, scale_factor=(5.0, 5.0))
    samples_dir = pj(cfg["session_dir"], "samples")
    if not pe(samples_dir):
        os.makedirs(samples_dir)
    tv.utils.save_image(xhat, pj(samples_dir,"%03d.png" % epoch))

def train(m_gen, m_disc, train_loader, optimizers, cfg):
    m_gen.apply(weights_init)
    m_disc.apply(weights_init)
    cudev = cfg["cuda"]
    batch_size = cfg["batch_size"]
    optD,optG = optimizers
    eps = 1e-6
    d_criterion = lambda yhat,y : -torch.mean( y*torch.log(yhat+eps) \
            + (1-y)*torch.log(1-yhat+eps) )
    g_criterion = lambda yhat : -torch.mean( torch.log(yhat+eps) )
    tboard_dir = pj(cfg["session_dir"], "tensorboard")
    if not pe(tboard_dir): os.makedirs(tboard_dir)
    writer = SummaryWriter( pj(cfg["session_dir"], "tensorboard") )
    models_dir = pj(cfg["session_dir"], "models")
    if not pe(models_dir): os.makedirs(models_dir)
    num_batches = len(train_loader) // batch_size
    for epoch in range(cfg["num_epochs"]):
        for i,(real_x,_) in enumerate(train_loader):
            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)
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

            d_loss.backward(retain_graph=True)
            optD.step()

            optG.zero_grad()
            g_loss = g_criterion(m_disc(fake_x))
            g_loss.backward()
            optG.step()

            writer.add_scalars("Loss", {"Generator" : g_loss.item(),
                "Discriminator/Real" : d_real_loss.item(),
                "Discriminator/Fake" : d_fake_loss.item()}, epoch*num_batches+1)

        logging.info("Epoch %d: GLoss: %.4f, DLossReal: %.4f, DLossFake: %.4f" \
                % (epoch, g_loss.item(), d_real_loss.item(),d_fake_loss.item()))
            
        save_sample_images(m_gen, epoch, cfg)
        torch.save(m_gen.state_dict(), pj(models_dir, "generator_%04d.pkl" \
                % (epoch)))
        torch.save(m_disc.state_dict(), pj(models_dir, "discriminator_%04d.pkl"\
                % (epoch)))
        
# Taken from pytorch DCGAN tutorial
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# TODO put in utils.py
def z_sampler(batch_size, z_dim, cudev):
    if cudev >= 0:
        z = torch.cuda.FloatTensor(batch_size, z_dim).normal_(0.0,1.0)
    else:
        z = torch.FloatTensor(batch_size, z_dim).normal_(0.0,1.0)
    return z


def main(args):
    cfg = vars(args)
    cfg["session_dir"] = create_session_dir("./sessions")
    m_gen = Generator(z_dim=cfg["z_dim"], num_layers=cfg["num_layers"],
            num_base_chans=cfg["num_base_chans"])
    m_disc = Discriminator(num_base_chans=cfg["num_base_chans"],
            num_layers=cfg["num_layers"]-1)
    init_session_log(cfg, "w")
    train_loader = get_loader(cfg)
    make_first_batch(train_loader, cfg)

    cudev = cfg["cuda"]
    if cudev >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA device specified but CUDA not available")
    if cudev >= 0:
        m_gen.cuda(cudev)
        m_disc.cuda(cudev)

    optD = torch.optim.Adam(m_disc.parameters(), lr=cfg["lr_d"],
            betas=(0.5, 0.999))
    optG = torch.optim.Adam(m_gen.parameters(), lr=cfg["lr_g"],
            betas=(0.5, 0.999))

    train(m_gen, m_disc, train_loader, (optD,optG), cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--celeba-path", type=str,
            default=pj("./data/celebA/celebA"))

    # Model
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--num-base-chans", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=100,
        help="Number of latent space units")

    # Training
    parser.add_argument("--lr-d", type=float, default=0.0001,
            help="Model learning rate")
    parser.add_argument("--lr-g", type=float, default=0.001,
            help="Model learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
            help="Momentum parameter for the SGD optimizer")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)

    # Hardware/OS
    parser.add_argument("--cuda", type=int, default=0,
            help="Cuda device number, select -1 for cpu")
    parser.add_argument("--num-workers", type=int, default=4,
        help="Number of worker threads to use loading data")

    args = parser.parse_args()
    main(args)
