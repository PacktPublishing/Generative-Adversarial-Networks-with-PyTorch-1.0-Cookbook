"""
This program runs inference using the Cifar-10 Generator we trained.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision as tv

from make_conv_layers import DeconvLayers

pe = os.path.exists
pj = os.path.join

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import z_sampler

class Generator(DeconvLayers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = x.view((-1, x.shape[1], 1, 1))
        x = super().forward(x)
        x = torch.tanh(x)
        return x


def main(args):
    cfg = vars(args)
    m_gen = Generator(z_dim=cfg["z_dim"], num_layers=4, num_base_chans=32)
    m_gen.load_state_dict( torch.load(cfg["model_path"]) )
    cudev = cfg["cuda"]
    if cudev >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA device specified but CUDA not available")
    if cudev >= 0:
        m_gen.cuda(cudev)
    ct = 0
    for b in range(cfg["num_batches"]):
        xhats = m_gen( z_sampler(cfg["batch_size"], cfg["z_dim"], cudev) )
        for xhat in xhats:
            tv.utils.save_image( xhat, pj(cfg["output_dir"],
                "test_%05d.png" % ct) )
            ct += 1
    print("Generated %d fake images to %s" % (ct, cfg["output_dir"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp", "--model-path", dest="model_path", type=str,
            required=True)
    parser.add_argument("-o", "--output-dir", type=str,
            default="./data/cifar10-fake")
    parser.add_argument("--cuda", type=int, default=-1,
            help="Cuda device number, select -1 for cpu")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("-n", "--num-batches", type=int, default=100)
    parser.add_argument("--z-dim", type=int, default=100,
        help="Number of latent space units")
    args = parser.parse_args()
    main(args)


