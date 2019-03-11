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


def check_paths(cfg):
    cfg["dataset_1"] = os.path.abspath(cfg["dataset_1"])
    cfg["dataset_2"] = os.path.abspath(cfg["dataset_2"])
    if not pe(cfg["dataset_1"]) or not pe(cfg["dataset_2"]):
        raise RuntimeError("Invalid path supplied")

def get_mean_and_sd(path, model, cfg):
    dataset = ImageFolder(path, cfg["ext"])
    data_loader = DataLoader(dataset, batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"], shuffle=False)
    m,s = calculate_activation_statistics(data_loader, model, cfg["cuda"])
    return m, s

def get_inception():
    inception = InceptionV3()
    return inception

#def compute_features(model, data_loader, gpu_device=0, make_chip_list=True):
def calculate_activation_statistics(data_loader, model, cuda=False):
    feats = compute_features(model, data_loader, cuda, make_chip_list=False)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def calculate_fid(cfg):
    model = get_inception()
    cudev = cfg["cuda"]
    if cudev >= 0:
        model.cuda(cudev)

    m1,s1 = get_mean_and_sd(cfg["dataset_1"], model, cfg)
    m2,s2 = get_mean_and_sd(cfg["dataset_2"], model, cfg)
    print(m1, s1, m2, s2)
#    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main(args):
    cfg = vars(args)
    fid = calculate_fid(cfg)
    print("FID: ", fid)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", "--dataset-1", dest="dataset_1", type=str,
            required=True)
    parser.add_argument("--d2", "--dataset-2", dest="dataset_2", type=str,
            required=True)
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

