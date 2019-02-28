"""
A simple convolutional neural network to classify the images in the Cifar19
dataset
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_session_dir, init_session_log

pe = os.path.exists
pj = os.path.join


#    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                 padding=0, dilation=1, groups=1, bias=True):
class SimpleCNN(nn.Module):
    def __init__(self, input_size=(3,32,32), num_cats=10):
        super().__init__()
        self._input_size = input_size
        self._num_cats = num_cats

        input_ch = self._input_size[0]
        self._conv1 = nn.Conv2d(input_ch, 32, kernel_size=5, stride=2,
                bias=True)
        self._bn1 = nn.BatchNorm2d(32)
        self._conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2,
                bias=True)
        self._bn2 = nn.BatchNorm2d(64)
        self._conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2,
                bias=True)
        self._bn3 = nn.BatchNorm2d(64)
        self._fc = nn.Linear(64, self._num_cats, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self._conv1(x)
        x = torch.relu( self._bn1(x) )
        x = self._conv2(x)
        x = torch.relu( self._bn2(x) )
        x = self._conv3(x)
        x = torch.relu( self._bn3(x) )
        x = x.view(batch_size, -1)
        x = torch.softmax( self._fc(x), dim=1 )
        return x

def get_loaders(cfg):
    train_transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor()
        ])
    train_loader = DataLoader(
            tv.datasets.CIFAR10("data", train=True, download=True,
                transform=train_transform),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"])
    test_loader = DataLoader(
            tv.datasets.CIFAR10("data", train=False, download=True,
                transform=tv.transforms.ToTensor()),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"])
    return train_loader,test_loader

def train(cnn, data_loaders, optimizer, cfg):
    cudev = cfg["cuda"]
    batch_size = cfg["batch_size"]
    train_loader,test_loader = data_loaders
    criterion = nn.CrossEntropyLoss()
    for epoch in range( cfg["num_epochs"] ):
        cnn.train()
        for i,(x,label) in enumerate(train_loader):
            if cudev >= 0:
                x = x.cuda(cudev)
                label = label.cuda(cudev)

            optimizer.zero_grad()
            yhat = cnn(x)
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()

            accuracy = 100.0 * torch.mean( \
                    (torch.argmax(yhat,dim=1) == label).float() )

            if i%100 == 99:
                logging.info("Training, Epoch %d, batch %d: Loss: %.4f, " \
                        "Accuracy: %.4f" % (epoch, i, loss.item(), accuracy))

        optimizer.zero_grad()
        running_loss = 0
        running_acc = 0
        ct = 0
        cnn.eval()
        for (x,label) in test_loader:
            if cudev >= 0:
                x = x.cuda(cudev)
                label = label.cuda(cudev)

            yhat = cnn(x)
            loss = criterion(yhat, label)
            accuracy = 100.0 * torch.mean( \
                    (torch.argmax(yhat,dim=1) == label).float() )
            running_loss += loss.item()
            running_acc += accuracy.item()
            ct += 1

        logging.info("TEST, Epoch %d: Loss: %.4f, Accuracy: %.4f" \
                % (epoch, running_loss/ct, running_acc/ct))


def main(args):
    cfg = vars(args)
    cfg["session_dir"] = create_session_dir("./sessions")
    start_epoch = 0
    filemode = "w"
    init_session_log(cfg, filemode)
    train_loader,test_loader = get_loaders(cfg)
    cudev = cfg["cuda"]
    if cudev >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA device specified but CUDA not available")
    cnn = SimpleCNN()
    if cudev >= 0:
        cnn = cnn.cuda(cudev)
    optimizer = torch.optim.SGD([{"params" : cnn.parameters()}], lr=cfg["lr"],
        momentum=cfg["momentum"])
    train(cnn, (train_loader,test_loader), optimizer, cfg)

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
    parser.add_argument("--lr", type=float, default=0.001,
            help="Model learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
            help="Momentum parameter for the SGD optimizer")
    parser.add_argument("--resume-path", type=str, default="",
            help="Path to directory with saved models")
    args = parser.parse_args()
    main(args)

