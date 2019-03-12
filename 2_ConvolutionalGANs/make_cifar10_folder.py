"""
The purpose of this simple script is to create a folder of Cifar-10 images. This
will make it easier to compare them to our GAN output.
"""

import argparse
import os
import torch
import torchvision as tv
from torch.utils.data import DataLoader

pe = os.path.exists
pj = os.path.join


def main(args):
    cfg = vars(args)
    loader = DataLoader(
            tv.datasets.CIFAR10("./data", train=True, download=True,
                transform=tv.transforms.ToTensor()),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"])
    ct = 0
    for batch,_ in loader:
        for x in batch:
            if ct==cfg["num_images"]:
                break
            tv.utils.save_image(x, pj(cfg["output_path"], "real_%05d.png" % ct))
            ct += 1

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str,
            default="./data/cifar10-real")
    parser.add_argument("-n", "--num-images", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=4,
        help="Number of worker threads to use loading data")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    main(args)

