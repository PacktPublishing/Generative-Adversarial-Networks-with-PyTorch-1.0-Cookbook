"""
This program runs inference using a trained generator.  I.e., it generates
samples.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision as tv

from generator import Generator, z_sampler

pe = os.path.exists
pj = os.path.join


def save_sample_images(m_gen, batch_num, cfg):
    z = z_sampler(cfg["batch_size"], cfg["z_dim"], cfg["cuda"])
    xhat = m_gen(z)
    xhat = F.interpolate(xhat, scale_factor=(5.0, 5.0))
    tv.utils.save_image(xhat, "samples/test_%03d.png" % batch_num)

def main(args):
    cfg = vars(args)
    m_gen = Generator(cfg["z_dim"])
    m_gen.load_state_dict( torch.load(cfg["model_path"]) )
    cudev = cfg["cuda"]
    if cudev >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA device specified but CUDA not available")
    if cudev >= 0:
        m_gen.cuda(cudev)
    for b in range(cfg["num_batches"]):
        save_sample_images(m_gen, b, cfg)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp", "--model-path", dest="model_path", type=str,
            required=True)
    parser.add_argument("--cuda", type=int, default=-1,
            help="Cuda device number, select -1 for cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("-n", "--num-batches", type=int, default=5)
    parser.add_argument("--z-dim", type=int, default=100,
        help="Number of latent space units")
    args = parser.parse_args()
    main(args)


