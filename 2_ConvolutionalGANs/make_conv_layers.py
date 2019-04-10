"""
This module comprises a stack of convolutional layers of arbitrary depth
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class ConvLayers(nn.Module):
    def __init__(self, num_layers=4, num_base_chans=16, kernel_size=4,
            stride=2, is_transpose=False, debug=False):
        super().__init__()
        self._debug = debug
        self._is_transpose = is_transpose
        self._kernel_size = kernel_size
        self._layers = None
        self._num_layers = num_layers
        self._num_base_chans = num_base_chans
        self._stride = stride

        self._layers = self._make_layers()

    def forward(self, x):
        if self._debug:
            print("ConvLayers forward:")
            for i,layer in enumerate(self._layers):
                logging.info("Layer %d shape in: %s" % (i, x.shape))
                x = layer(x)
                logging.info("\tshape out: %s" % repr(x.shape))
            return x
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
        return nn.Sequential( *layers )

class DeconvLayers(nn.Module):
    def __init__(self, num_layers=4, z_dim=100, num_base_chans=16,
            kernel_size=4, stride=2, debug=False):
        super().__init__()
        self._debug = debug
        self._kernel_size = kernel_size
        self._layers = None
        self._num_layers = num_layers
        self._num_base_chans = num_base_chans
        self._stride = stride
        self._z_dim = z_dim

        self._layers = self._make_layers()

    def forward(self, x):
        if self._debug:
            print("DeconvLayers forward:")
            for i,layer in enumerate(self._layers):
                logging.info("Layer %d shape in: %s" % (i, x.shape))
                x = layer(x)
                logging.info("\tshape out: %s" % repr(x.shape))
            return x
        return self._layers(x)

    def _make_layers(self):
        layers = []
        c_in = self._z_dim
        c_out = self._num_base_chans * (2**(self._num_layers-1))
        for i in range(self._num_layers):
            if i==0:
                layers.append( nn.ConvTranspose2d(c_in, c_out,
                    kernel_size=self._kernel_size, padding=0, stride=1,
                    bias=False) )
            else:
                layers.append( nn.ConvTranspose2d(c_in, c_out,
                    kernel_size=self._kernel_size, padding=1,
                    stride=self._stride, bias=False) )
            if i<self._num_layers - 1:
                layers.append( nn.BatchNorm2d(c_out) )
                layers.append( nn.ReLU(inplace=True) )
            c_in = c_out
            if i==self._num_layers - 2:
                c_out = 3
            else:
                c_out = c_out // 2
        return nn.Sequential( *layers )


def _test_main(args):
    if args.test == "ConvLayers":
        print("Creating layers suitable for a Discriminator")
        net = ConvLayers(num_layers=args.num_layers,
                num_base_chans=args.num_base_chans,
                kernel_size=args.kernel_size,
                stride=args.stride,
                is_transpose=args.is_transpose)
        sz = args.test_input_size
        x = torch.FloatTensor(1, 3, sz, sz).normal_(0,1)
    else:
        print("Creating layers suitable for a Generator")
        net = DeconvLayers(num_layers=args.num_layers,
                num_base_chans=args.num_base_chans,
                z_dim=args.z_dim,
                kernel_size=args.kernel_size,
                stride=args.stride)
        x = torch.FloatTensor(1, args.z_dim, 1, 1).normal_(0,1)
    print(net)
    print("Input shape: %s" % repr(x.shape))
    y = net(x)
    print("Output shape: %s" % repr(y.shape))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="ConvLayers",
            choices=["ConvLayers", "DeconvLayers"])
    parser.add_argument("--test-input-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--num-base-chans", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--is_transpose", action="store_true")
    args = parser.parse_args()
    _test_main(args)

