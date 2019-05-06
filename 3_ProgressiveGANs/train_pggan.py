"""
This is an implentation of Karras et al. 2015 which will train on Celeb-A and
generate images accordingly.
"""

import argparse
import logging
import numpy as np
import os
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

from pglayers import ConvLayers, DeconvLayers
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
        self._last_conv = None

        c_out = self._num_base_chans*( 2**(self._num_layers-1) )
        c_out += 1 # For minibatch std dev
        self._last_conv = nn.Conv2d(c_out, 1, kernel_size=self._kernel_size,
                stride=1, padding=0)

    def forward(self, x, minibatch_sd):
        x = super().forward(x)
        mbsd = torch.ones((x.shape[0],1,*x.shape[2:])) * minibatch_sd
        x = self._last_conv( torch.cat((x,mbsd), dim=1) )
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


def get_loader(num_layers, batch_size, cfg):
    input_size = 2**(num_layers + 1)
    print("Input size: %d" % input_size)
    train_transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(brightness=0.25, contrast=0.25,
            saturation=0.25, hue=0.025),
        tv.transforms.Resize(input_size),
        tv.transforms.CenterCrop(input_size),
        tv.transforms.ToTensor()
        ])
    celeba_dataset = ImageFolder(cfg["celeba_path"], transform=train_transform,
            ext=".jpg")
    train_loader = DataLoader(dataset=celeba_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg["num_workers"])
    return train_loader

def make_first_batch(data_loader, cfg):
    fb_dir = pj(cfg["session_dir"], "first_batch")
    if not pe(fb_dir):
        os.makedirs(fb_dir)
    cudev = cfg["cuda"]
    for real_x,_ in data_loader:
        if cudev >= 0:
            real_x = real_x.cuda(cudev)
        break
    real_x = real_x[:32]
    sz = real_x.shape[2]
    tv.utils.save_image(real_x, pj(fb_dir, "first_real_batch_%03d.png" % sz))

def minibatch_std_dev(x):
    sd = torch.std(x, dim=0)
    return torch.mean(sd)

def save_sample_images(m_gen, scale_i, epoch, cfg):
    z = z_sampler(32, cfg["z_dim"], cfg["cuda"])
    xhat = m_gen(z)
    xhat = F.interpolate(xhat, scale_factor=(5.0, 5.0))
    samples_dir = pj(cfg["session_dir"], "samples")
    if not pe(samples_dir):
        os.makedirs(samples_dir)
    tv.utils.save_image(xhat, pj(samples_dir,"%d_%03d.png" % (scale_i, epoch)))

def train(cfg):
    cudev = cfg["cuda"]
    if cudev >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA device specified but CUDA not available")

    num_layers = 2
    m_gen = Generator(z_dim=cfg["z_dim"], num_layers=num_layers,
            num_base_chans=cfg["num_base_chans"])
    m_disc = Discriminator(num_base_chans=cfg["num_base_chans"],
            num_layers=num_layers-1)
    if cudev >= 0:
        m_gen.cuda(cudev)
        m_disc.cuda(cudev)
    logging.info("Generator:\n")
    logging.info(str(m_gen))
    logging.info("\n\n")
    logging.info("Discriminator:\n")
    logging.info(str(m_disc))
    logging.info("Scale: 0")
    logging.info("\n\n")

    betas = (cfg["beta1"], cfg["beta2"])
    optD = torch.optim.Adam(m_disc.parameters(), lr=cfg["lr_d"], betas=betas,
            eps=cfg["epsilon"])
    optG = torch.optim.Adam(m_gen.parameters(), lr=cfg["lr_g"], betas=betas,
            eps=cfg["epsilon"])
    eps = 1e-6
    d_criterion = nn.BCELoss()
    g_criterion = nn.BCELoss()
    tboard_dir = pj(cfg["session_dir"], "tensorboard")
    if not pe(tboard_dir): os.makedirs(tboard_dir)
    writer = SummaryWriter( pj(cfg["session_dir"], "tensorboard") )
    models_dir = pj(cfg["session_dir"], "models")
    if not pe(models_dir): os.makedirs(models_dir)

    epochs_by_scale = np.ones(5).astype(int) * cfg["num_epochs"]
    num_scales = len(epochs_by_scale)
    for scale_i in range(num_scales):
        print("############# New scale #############")
        if scale_i == 0:
            m_gen.apply(weights_init)
            m_disc.apply(weights_init)
        else:
            m_gen.add_layer()
            m_disc.add_layer()
            if cudev >= 0:
                m_gen.cuda(cudev)
                m_disc.cuda(cudev)
            num_layers += 1

        batch_size = cfg["final_batch_size"] * 2**(num_scales-scale_i-1)
        data_loader = get_loader(num_layers, batch_size, cfg)
        make_first_batch(data_loader, cfg)
        num_batches = len(data_loader) // batch_size

        num_epochs = epochs_by_scale[scale_i]
        batches_in_scale = len(data_loader) * num_epochs
        global_ct = 0
        for epoch in range(num_epochs):
            for i,(real_x,_) in enumerate(data_loader):
                alpha = global_ct / batches_in_scale
                global_ct += 1
                m_gen.set_alpha(alpha)
                m_disc.set_alpha(alpha)
                batch_size = real_x.shape[0]
                real_labels = torch.ones(batch_size)
                fake_labels = torch.zeros(batch_size)
                if cudev >= 0:
                    real_x = real_x.cuda(cudev)
                    real_labels = real_labels.cuda(cudev)
                    fake_labels = fake_labels.cuda(cudev)
                    labels = torch.cat((torch.ones(batch_size), 
                        torch.zeros(batch_size))).cuda(cudev)
                z = z_sampler(batch_size, cfg["z_dim"], cudev)
                real_std_dev = minibatch_std_dev(real_x)

                if epoch==0 and i==0 and cfg["debug"]:
                    m_gen._debug = True
                    m_disc._debug = True
                else:
                    m_gen._debug = False
                    m_disc._debug = False

                optD.zero_grad()
                fake_x = m_gen(z)
                fake_std_dev = minibatch_std_dev(fake_x)
                d_fake_loss = d_criterion(m_disc(fake_x, fake_std_dev),
                        fake_labels)
                d_real_loss = d_criterion(m_disc(real_x, real_std_dev),
                        real_labels)
                d_loss = d_fake_loss + d_real_loss

                d_loss.backward(retain_graph=True)
                optD.step()

                optG.zero_grad()
                g_loss = g_criterion(m_disc(fake_x, fake_std_dev),
                        real_labels)
                g_loss.backward()
                optG.step()

                writer.add_scalars("Loss", {"Generator" : g_loss.item(),
                    "Discriminator/Real" : d_real_loss.item(),
                    "Discriminator/Fake" : d_fake_loss.item()},
                    epoch*num_batches+1)

#                if global_ct-1 % 10 == 0:
                print("Alpha: %f" % alpha)

            logging.info("Epoch %d: GLoss: %.4f, DLossReal: %.4f, DLossFake: " \
                    "%.4f" % (epoch, g_loss.item(), d_real_loss.item(),
                        d_fake_loss.item()))
                
            save_sample_images(m_gen, scale_i, epoch, cfg)
            torch.save(m_gen.state_dict(), pj(models_dir, "generator_%04d.pkl" \
                    % (epoch)))
            torch.save(m_disc.state_dict(), pj(models_dir,
                "discriminator_%04d.pkl" % (epoch)))
        
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
    init_session_log(cfg)

    train(cfg)

def _test_models(args):
    cfg = vars(args)
    cudev = cfg["cuda"]
    num_layers = 2
    if cfg["test_model"] == "Discriminator" or cfg["test_model"] == "Disc":
        print("Creating layers suitable for a Discriminator")
        net = Discriminator(num_base_chans=cfg["num_base_chans"],
                num_layers=num_layers-1, debug=True)
        sz = cfg["test_input_size"]
        x = torch.FloatTensor(1, 3, sz, sz).normal_(0,1)
    else:
        print("Creating layers suitable for a Generator")
        net = Generator(z_dim=cfg["z_dim"], num_layers=num_layers,
                num_base_chans=cfg["num_base_chans"], debug=True)
        x = torch.FloatTensor(1, cfg["z_dim"], 1, 1).normal_(0,1)
    if cudev>=0:
        net = net.cuda(cudev)
        x = x.cuda(cudev)
    print(net)
    print("Input shape: %s" % repr(x.shape))
    y = net(x)
    print("Output shape: %s" % repr(y.shape))

    for _ in range(3):
        net.add_layer()
        print(net)
        if cfg["test_model"] == "Discriminator" or cfg["test_model"] == "Disc":
            sz = x.shape[2] * 2
            x = torch.FloatTensor(1,3,sz,sz).normal_(0,1)
        if cudev>=0:
            net = net.cuda(cudev)
            x = x.cuda(cudev)
        print("Input shape: %s" % repr(x.shape))
        y = net(x)
        print("Output shape: %s" % repr(y.shape))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Test models
    parser.add_argument("--test-model", type=str, default=None,
            choices=["Gen", "Generator", "Disc", "Discriminator", None])
    parser.add_argument("--test-input-size", type=int, default=8)

    # Dataset
    parser.add_argument("--celeba-path", type=str,
            default=pj("./data/celebA/celebA"))

    # Model
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-base-chans", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=100,
        help="Number of latent space units")

    # Training
    parser.add_argument("--lr-d", type=float, default=0.0001,
            help="Model learning rate")
    parser.add_argument("--lr-g", type=float, default=0.001,
            help="Model learning rate")
    parser.add_argument("--beta1", type=float, default=0.0,
            help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.99,
            help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--epsilon", type=float, default=1e-8,
            help="epsilon parameter for the Adam optimizer")
    parser.add_argument("--momentum", type=float, default=0.9,
            help="Momentum parameter for the SGD optimizer")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--final-batch-size", type=int, default=256)
    parser.add_argument("--debug", action="store_true")

    # Hardware/OS
    parser.add_argument("--cuda", type=int, default=0,
            help="Cuda device number, select -1 for cpu")
    parser.add_argument("--num-workers", type=int, default=4,
        help="Number of worker threads to use loading data")

    args = parser.parse_args()
    if args.test_model is not None:
        _test_models(args)
    else:
        main(args)

