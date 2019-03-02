"""
This module comprises a stack of convolutional layers of arbitrary depth
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class ConvLayers(nn.Module):
    def __init__(self, num_layers=4, num_base_chans=16, kernel_size=3,
            stride=2, is_transpose=False):
        super().__init__()
        self._is_transpose = is_transpose
        self._kernel_size = kernel_size
        self._layers = None
        self._num_layers = num_layers
        self._num_base_chans = num_base_chans
        self._stride = stride

        self._layers = self._make_layers()

    def forward(self, x):
        return self._layers(x)

    def _make_layers(self):
        layers = []
        c_in = 3
        c_out = self._num_base_chans
        for i in range(self._num_layers):
            layers.append( nn.Conv2d(c_in, c_out, kernel_size=self._kernel_size,
                padding=1, stride=self._stride, bias=False) )
            layers.append( nn.BatchNorm2d(c_out) )
            layers.append( nn.LeakyReLU(0.2) )
            c_in = c_out
            c_out = self._num_base_chans * (2**(i+1))
        print(len(layers))
        return nn.Sequential( *layers )


def _test_main(args):
    conv_net = ConvLayers(num_layers=args.num_layers,
            num_base_chans=args.num_base_chans,
            kernel_size=args.kernel_size,
            stride=args.stride,
            is_transpose=args.is_transpose)
    print(conv_net)
    x = torch.FloatTensor(1, 3, 64, 64).normal_(0,1)
    print("Input shape: %s" % repr(x.shape))
    y = conv_net(x)
    print("Output shape: %s" % repr(y.shape))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-base-chans", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--is_transpose", action="store_true")
    args = parser.parse_args()
    _test_main(args)

