"""
This function calculates the Frechet Inception Distance between two datasets.
"""

import argparse
import os
import numpy as np
import sys
from scipy import linalg
from scipy.misc import imread
from PIL import Image

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compute_features

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

pe = os.path.exists
pj = os.path.join


class ImageFolder(Dataset):
    def __init__(self, data_path, ext=".png"):
        self._images = [pj(data_path,f) for f in os.listdir(data_path) \
                if f.endswith(ext)]
        self._transform = tv.transforms.ToTensor()

    def __getitem__(self, index):
        return self._transform( Image.open(self._images[index]) ), -1

    def __len__(self):
        return len(self._images)

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = tv.models.inception_v3(pretrained=True)
        self._layers = [
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))]
        self._model = nn.Sequential(*self._layers)

    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear",
                align_corners=False)
        x = 2.0*x - 1.0
        for layer in self._layers:
            x = layer(x)
        return x

    def get_features(self, x):
        return self.forward(x)


def calculate_fid(cfg):
    model = get_inception()
    cudev = cfg["cuda"]
    if cudev >= 0:
        model.cuda(cudev)
    m1,cov1 = get_mean_and_cov(cfg["dataset_1"], model, cfg)
    m2,cov2 = get_mean_and_cov(cfg["dataset_2"], model, cfg)
    fid_value = calculate_frechet(m1, cov1, m2, cov2)
    return fid_value

# Lucic et al. 2017
def calculate_frechet(mu1, sigma1, mu2, sigma2):
    dmu = mu1 - mu2
    cov_mean,_ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    frechet = dmu.dot(dmu) + np.trace(sigma1 + sigma2 - 2*cov_mean)
    return frechet

def check_paths(cfg):
    cfg["dataset_1"] = os.path.abspath(cfg["dataset_1"])
    cfg["dataset_2"] = os.path.abspath(cfg["dataset_2"])
    if not pe(cfg["dataset_1"]) or not pe(cfg["dataset_2"]):
        raise RuntimeError("Invalid path supplied")

def get_mean_and_cov(path, model, cfg):
    dataset = ImageFolder(path, cfg["ext"])
    data_loader = DataLoader(dataset, batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"], shuffle=False)
    feats = compute_features(model, data_loader, cfg["cuda"],
            make_chip_list=False)
    mu = np.mean(feats, axis=0)
    cov = np.cov(feats, rowvar=False)
    return mu,cov

def get_inception():
    inception = InceptionV3()
    return inception


def main(args):
    cfg = vars(args)
    check_paths(cfg)
    fid = calculate_fid(cfg)
    print("FID: ", fid)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", "--dataset-1", dest="dataset_1", type=str,
            default="./data/cifar10-real")
    parser.add_argument("--d2", "--dataset-2", dest="dataset_2", type=str,
            default="./data/cifar10-fake")
    parser.add_argument("--cuda", type=int, default=0,
            help="Cuda device number, select -1 for cpu")
    parser.add_argument("--num-workers", type=int, default=4,
        help="Number of worker threads to use loading data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ext", "--extension", dest="ext", type=str,
            default=".png")
    parser.add_argument("--incep-feat-dim", type=int, default=2048,
            help="Size of Inception feature vectors to use")
    args = parser.parse_args()
    main(args)

