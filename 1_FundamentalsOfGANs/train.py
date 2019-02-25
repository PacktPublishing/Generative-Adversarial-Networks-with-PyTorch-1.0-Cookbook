"""
This is a very simple, self-contained GAN which trains on MNIST and generates
samples accordingly.
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from generator import Generator, z_sampler
from discriminator import Discriminator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_session_dir, init_session_log

pe = os.path.exists
pj = os.path.join


def get_loader(cfg):
    train_loader = DataLoader(
            tv.datasets.MNIST("data", train=True, download=True,
                transform=tv.transforms.ToTensor()),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"])
    return train_loader

def load_models(m_gen, m_disc, cfg):
    model_list = sorted( os.listdir( cfg["resume_path"] ) )
    gen_name = [x for x in model_list if x.startswith("gen")][-1]
    disc_name = [x for x in model_list if x.startswith("disc")][-1]
    m_gen.load_state_dict( torch.load( pj(cfg["resume_path"], gen_name) ) )
    m_disc.load_state_dict( torch.load( pj(cfg["resume_path"], disc_name) ) )
    pos = gen_name.index("_")+1
    start_epoch = int( gen_name[pos:-4] )
    return start_epoch

def save_sample_images(m_gen, epoch, cfg):
    z = z_sampler(cfg["batch_size"], cfg["z_dim"], cfg["cuda"])
    xhat = m_gen(z)
    xhat = F.interpolate(xhat, scale_factor=(5.0, 5.0))
    tv.utils.save_image(xhat, "samples/%03d.png" % epoch)

def train(m_gen, m_disc, train_loader, optimizers, cfg, start_epoch=0):
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
    num_epochs = len(train_loader) // batch_size
    for epoch in range(start_epoch, cfg["num_epochs"]):
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
                "Discriminator/Fake" : d_fake_loss.item()}, epoch*num_epochs+1)

        logging.info("Epoch %d: GLoss: %.4f, DLossReal: %.4f, DLossFake: %.4f" \
                % (epoch, g_loss.item(), d_real_loss.item(),d_fake_loss.item()))
            
        save_sample_images(m_gen, epoch, cfg)
        torch.save(m_gen.state_dict(), pj(models_dir, "generator_%04d.pkl" \
                % (epoch)))
        torch.save(m_disc.state_dict(), pj(models_dir, "discriminator_%04d.pkl"\
                % (epoch)))
        

def main(args):
    cfg = vars(args)
    cfg["session_dir"] = create_session_dir("./sessions")
    m_gen = Generator(cfg["z_dim"])
    m_disc = Discriminator()
    if len( cfg["resume_path"] ) > 0:
        cfg["session_dir"] = os.path.dirname( os.path.abspath(\
                cfg["resume_path"] ) )
        start_epoch = load_models(m_gen, m_disc, cfg)
        filemode = "a"
    else:
        start_epoch = 0
        filemode = "w"
    init_session_log(cfg, filemode)
    train_loader = get_loader(cfg)
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
    train(m_gen, m_disc, train_loader, (optD,optG), cfg, start_epoch)

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
    parser.add_argument("--resume-path", type=str, default="",
            help="Path to directory with saved models")
    args = parser.parse_args()
    main(args)
