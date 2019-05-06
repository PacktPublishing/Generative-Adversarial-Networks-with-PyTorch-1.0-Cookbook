"""
This module comprises a stack of convolutional layers of arbitrary depth, with
the ability to train progressively.
"""

import abc
import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class ProgNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._alpha = 1.0
        self._transition_layer = None

    @abc.abstractmethod
    def add_layer(self):
        raise NotImplementedError()

    def set_alpha(self, alpha):
        self._alpha = alpha


class ConvLayers(ProgNet):
    def __init__(self, num_layers=4, num_base_chans=16, kernel_size=4,
            stride=2, is_transpose=False, debug=False):
        super().__init__()
        self._avg_pool = nn.AvgPool2d(2)
        self._debug = debug
        self._is_transpose = is_transpose
        self._kernel_size = kernel_size
        self._layers = None
        self._num_layers = num_layers
        self._num_base_chans = num_base_chans
        self._stride = stride

        self._layers = self._make_layers()

    def add_layer(self):
        self._transition_layer = self._layers[0]
        for p in self._transition_layer.parameters():
            p.requires_grad = False

        old_layers = self._layers[1:]
        c_in = old_layers[0].in_channels // 2
        c_out = old_layers[0].in_channels

        self._layers = nn.Sequential()
        self._layers.add_module( "0", self._get_fromRGB(c_in) )
        self._layers.add_module( "1", nn.Conv2d(c_in, c_out,
            kernel_size=self._kernel_size, padding=1, stride=self._stride,
            bias=False) )
        self._layers.add_module( "2", nn.BatchNorm2d(c_out) )
        self._layers.add_module( "3", nn.ReLU(inplace=True) )

        N = len(old_layers)
        for i in range(len(old_layers)):
            self._layers.add_module( str(i+4), old_layers[i] )

        self._num_layers += 3

    def forward(self, x):
        if self._debug: print("ConvLayers forward:")
        if self._transition_layer is not None:
            tx = self._avg_pool(x)
            tx = self._transition_layer(tx)

            for i,layer in enumerate( self._layers[:4] ):
                if self._debug:
                    logging.info("Layer %d shape in: %s" % (i, x.shape))
                x = layer(x)
                if self._debug:
                    logging.info("\tshape out: %s" % repr(x.shape))

            x = (1.0 - self._alpha)*tx + self._alpha*x

            for layer in self._layers[4:]:
                if self._debug:
                    logging.info("Layer %d shape in: %s" % (i, x.shape))
                x = layer(x)
                if self._debug:
                    logging.info("\tshape out: %s" % repr(x.shape))
        else:
            x = self._layers(x)

        return x

    # This assumes any preexisting fromRGB layer has already been removed
    def _get_fromRGB(self, c_out):
        return nn.Conv2d(3, c_out, kernel_size=1, padding=0, stride=1,
                bias=False)

    def _make_layers(self):
        layers = []
        start_chans = self._num_base_chans // (2**self._num_layers)
        layers.append( self._get_fromRGB(start_chans) )

        c_in = start_chans
        c_out = self._num_base_chans
        for i in range(self._num_layers):
            layers.append( nn.Conv2d(c_in, c_out, kernel_size=self._kernel_size,
                padding=1, stride=self._stride, bias=False) )
            layers.append( nn.BatchNorm2d(c_out) )
            layers.append( nn.LeakyReLU(0.2) )
            c_in = c_out
            c_out =  c_out * 2

        return nn.Sequential( *layers )

class DeconvLayers(ProgNet):
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

    def add_layer(self):
        self._transition_layer = self._layers[-1]
        for p in self._transition_layer.parameters():
            p.requires_grad = False

        self._layers = self._layers[:-1]
        c_in = self._layers[-1].in_channels // 2
        N = len(self._layers)
        self._layers.add_module( str(N), nn.BatchNorm2d(c_in) )
        self._layers.add_module( str(N+1), nn.ReLU(inplace=True) )

        c_out = c_in // 2
        self._layers.add_module( str(N+2), nn.ConvTranspose2d(c_in, c_out,
            kernel_size=self._kernel_size, padding=1, stride=self._stride,
            bias=False) )
        self._add_toRGB(self._layers, c_out)
        self._num_layers += 3

    def forward(self, x):
        if self._debug: print("DeconvLayers forward:")
        for i,layer in enumerate(self._layers[:-4]):
            if self._debug: logging.info("Layer %d shape in: %s" % (i, x.shape))
            x = layer(x)
            if self._debug: logging.info("\tshape out: %s" % repr(x.shape))

        N = len(self._layers)
        if self._debug:
            logging.info("Layer %d shape in: %s" % (N-1, x.shape))
        if self._transition_layer is not None:
            new_sz = x.shape[2] * 2
            tx = F.interpolate(x, size=(new_sz, new_sz))
            tx = self._transition_layer(tx)
            for layer in self._layers[-4 : -1]:
                x = layer(x)
            x = (1.0 - self._alpha)*tx + self._alpha*self._layers[-1](x)
        else:
            for layer in self._layers[-4 : -1]:
                x = layer(x)
            x = self._layers[-1](x)
        if self._debug: logging.info("\tshape out: %s" % repr(x.shape))

        return x

    def _add_toRGB(self, layers, c_in):
        N = len(layers)
        layers.add_module( str(N), nn.ConvTranspose2d(c_in, 3, 1, padding=0,
            stride=1, bias=False) )

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
            if i<self._num_layers - 1: # TODO
                layers.append( nn.BatchNorm2d(c_out) )
                layers.append( nn.ReLU(inplace=True) )
            c_in = c_out
            c_out = c_out // 2

        layers = nn.Sequential( *layers )
        self._add_toRGB(layers, c_in)

        return layers


def _test_main(args):
    cfg = vars(args)
    logging.basicConfig(level=logging.DEBUG)
    if args.test == "ConvLayers":
        print("Creating layers suitable for a Discriminator")
        net = ConvLayers(num_layers=args.num_layers,
                num_base_chans=args.num_base_chans,
                kernel_size=args.kernel_size,
                stride=args.stride,
                is_transpose=args.is_transpose,
                debug=True)
        sz = args.test_input_size
        x = torch.FloatTensor(1, 3, sz, sz).normal_(0,1)
    else:
        print("Creating layers suitable for a Generator")
        net = DeconvLayers(num_layers=args.num_layers,
                num_base_chans=args.num_base_chans,
                z_dim=args.z_dim,
                kernel_size=args.kernel_size,
                stride=args.stride,
                debug=True)
        x = torch.FloatTensor(1, args.z_dim, 1, 1).normal_(0,1)
    print(net)
    print("Input shape: %s" % repr(x.shape))
    y = net(x)
    print("Output shape: %s" % repr(y.shape))

    if cfg["progressive"]:
        for _ in range(3):
            net.add_layer()
            print(net)
            if args.test == "ConvLayers":
                sz = x.shape[2] * 2
                x = torch.FloatTensor(1,3,sz,sz).normal_(0,1)
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
    parser.add_argument("--pg", "--progressive", dest="progressive",
            action="store_true")
    args = parser.parse_args()
    _test_main(args)

