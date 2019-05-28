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
    def __init__(self, batch_norm=True, debug=False):
        super().__init__()
        self._alpha = 1.0
        self._batch_norm = batch_norm
        self._debug = debug
        self._transition_layer = None

    @abc.abstractmethod
    def add_layer(self):
        raise NotImplementedError()

    def set_alpha(self, alpha):
        self._alpha = alpha


class ConvLayers(ProgNet):
    def __init__(self, num_layers=4, num_max_chans=512, num_start_chans=16,
            kernel_size=3, stride=2, **kwargs):
        super().__init__(**kwargs)
        self._avg_pool = nn.AvgPool2d(2)
        self._current_channels = None
        self._kernel_size = kernel_size
        self._layers = None
        self._num_layers = num_layers
        self._num_max_chans = num_max_chans
        self._num_start_chans = num_start_chans
        self._stride = stride

        self._layers = self._make_layers()

    def add_layer(self):
        self._transition_layer = self._layers[0]
        for p in self._transition_layer.parameters():
            p.requires_grad = False

        old_layers = self._layers[1:]
        N = len(old_layers)
        c_in = self._current_channels
        c_out = c_in * 2

        ct = 0
        self._layers = nn.Sequential()
        self._layers.add_module( str(ct), self._get_fromRGB(c_in) )

        self._layers.add_module( "1", nn.Conv2d(c_in, c_out,
            kernel_size=self._kernel_size, padding=1, stride=self._stride,
            bias=False) )
        self._layers.add_module( "2", nn.BatchNorm2d(c_out) )
        self._layers.add_module( "3", nn.ReLU(inplace=True) )

        for i in range(len(old_layers)):
            self._layers.add_module( str(i+4), old_layers[i] )

        self._current_channels = c_out

    def forward(self, x):
        if self._debug: print("ConvLayers forward:")
        if self._debug: logging.info("x shape: %s" % repr(x.shape))
        if self._transition_layer is not None:
            tx = self._avg_pool(x)
            if self._debug: logging.info("tx shape: %s" % repr(tx.shape))
            tx = self._transition_layer(tx)
            if self._debug: logging.info("tx shape: %s" % repr(tx.shape))

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
        if self._debug: logging.info("Final x shape: %s" % repr(x.shape))

        return x

    def get_max_chans(self):
        return self._num_max_chans

    # This assumes any preexisting fromRGB layer has already been removed
    def _get_fromRGB(self, c_out):
        return nn.Conv2d(3, c_out, kernel_size=1, padding=0, stride=1,
                bias=False)

    def _make_layers(self):
        layers = []
        start_chans = self._num_start_chans
        layers.append( self._get_fromRGB(start_chans) )

        c_in = self._num_start_chans
#        c_out = self._num_max_chans
#        layers.append( nn.Conv2d(c_in, c_out, kernel_size=1,
#            padding=0, stride=1, bias=False) )
#        if self._batch_norm:
#            layers.append( nn.BatchNorm2d(c_out) )
#
#        c_in = self._num_max_chans
        c_out = self._num_max_chans
        layers.append( nn.Conv2d(c_in, c_out, kernel_size=self._kernel_size,
            padding=1, stride=self._stride, bias=False) )
        if self._batch_norm:
            layers.append( nn.BatchNorm2d(c_out) )
        layers.append( nn.LeakyReLU(0.2) )
        layers.append( nn.LeakyReLU(0.2) )

        c_in = c_out
        c_out = self._num_max_chans
        layers.append( nn.Conv2d(c_in, c_out, kernel_size=4, #self._kernel_size,
            padding=1, stride=self._stride, bias=False) )
        layers.append( nn.LeakyReLU(0.2) )
        self._current_channels = self._num_start_chans
       
        return nn.Sequential( *layers )

class DeconvLayers(ProgNet):
    def __init__(self, z_dim=100, kernel_size=4, stride=2, **kwargs):
        super().__init__(**kwargs)
        self._current_channels = None
        self._kernel_size = kernel_size
        self._layers = None
        self._prior_layer_ct = None
        self._stride = stride
        self._z_dim = z_dim

        self._layers = self._make_layers()

    def add_layer(self):
        self._transition_layer = self._layers[-1]
        for p in self._transition_layer.parameters():
            p.requires_grad = False

        self._layers = self._layers[:-1]
        c_in = self._current_channels
        N = len(self._layers)
        self._prior_layer_ct = N

        c_out = c_in // 2 # TODO Karras doesn't decimate until later
        self._layers.add_module( str(N), nn.ConvTranspose2d(c_in, c_out,
            kernel_size=self._kernel_size, padding=1, stride=self._stride,
            bias=False) )

        ct = 1
        self._layers.add_module( str(N+ct), nn.Conv2d(c_out, c_out,
            kernel_size=3, padding=1, stride=1, bias=False) )
        ct += 1
        if self._batch_norm:
            self._layers.add_module( str(N+ct), nn.BatchNorm2d(c_out) )
            ct += 1
        self._layers.add_module( str(N+ct), nn.ReLU(inplace=True) )
        ct +=1

        self._layers.add_module( str(N+ct), nn.Conv2d(c_out, c_out,
            kernel_size=3, padding=1, stride=1, bias=False) )
        ct += 1
        if self._batch_norm:
            self._layers.add_module( str(N+ct), nn.BatchNorm2d(c_out) )
            ct += 1
        self._layers.add_module( str(N+ct), nn.ReLU(inplace=True) )
        ct +=1

        self._layers.add_module( str(N+ct), self._get_toRGB(c_out) )
        self._current_channels = c_out

    def forward(self, x):
        if self._debug: print("DeconvLayers forward:")
        if self._debug: logging.info("x shape: %s" % repr(x.shape))
        for i,layer in enumerate(self._layers[:self._prior_layer_ct]):
            if self._debug: logging.info("Layer %d shape in: %s" % (i, x.shape))
            x = layer(x)
            if self._debug: logging.info("\tshape out: %s" % repr(x.shape))

        N = len(self._layers)
        if self._transition_layer is not None:
            new_sz = x.shape[2] * 2
            tx = F.interpolate(x, size=(new_sz, new_sz))
            if self._debug: logging.info("tx shape: %s" % repr(tx.shape))
            tx = self._transition_layer(tx)
            if self._debug: logging.info("tx shape: %s" % repr(tx.shape))
            for i,layer in enumerate(self._layers[self._prior_layer_ct : -1]):
                if self._debug:
                    logging.info("Layer %d shape in: %s" \
                            % (N-self._prior_layer_ct+i, x.shape))
                x = layer(x)
                if self._debug:
                    logging.info("\tshape out: %s" % repr(x.shape))
            x = (1.0 - self._alpha)*tx + self._alpha*self._layers[-1](x)
        else:
            for i,layer in enumerate( self._layers[self._prior_layer_ct : -1] ):
                if self._debug:
                    logging.info("Layer %d shape in: %s" \
                            % (N-self._prior_layer_ct+i, x.shape))
                x = layer(x)
                if self._debug:
                    logging.info("\tshape out: %s" % repr(x.shape))
            x = self._layers[-1](x)
        if self._debug: logging.info("Final x shape: %s" % repr(x.shape))

        return x

    def _add_sublayer(self, c_in):
        return ct

    def _get_toRGB(self, c_in):
        return nn.Conv2d(c_in, 3, 1, padding=0, stride=1, bias=False)

    def _make_layers(self):
        layers = []
        c_in = self._z_dim
        layers.append( nn.ConvTranspose2d(c_in, c_in,
            kernel_size=self._kernel_size, padding=0, stride=1,
            bias=False) )
        if self._batch_norm:
            layers.append( nn.BatchNorm2d(c_in) )
        layers.append( nn.ReLU(inplace=True) )

        layers.append( nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, stride=1,
            bias=False) )
        if self._batch_norm:
            layers.append( nn.BatchNorm2d(c_in) )
        layers.append( nn.ReLU(inplace=True) )
        self._prior_layer_ct = len(layers)

        layers.append( self._get_toRGB(c_in) )
        layers = nn.Sequential( *layers )
        self._current_channels = c_in

        return layers

def _test_main(args):
    cfg = vars(args)
    logging.basicConfig(level=logging.DEBUG)
    if args.test == "ConvLayers":
        print("Creating layers suitable for a Discriminator")
        net = ConvLayers(num_layers=cfg["num_layers"],
                num_max_chans=cfg["num_max_chans"],
                kernel_size=cfg["kernel_size"],
                stride=cfg["stride"],
                debug=True)
        sz = args.test_input_size
        x = torch.FloatTensor(1, 3, sz, sz).normal_(0,1)
    else:
        print("Creating layers suitable for a Generator")
        net = DeconvLayers(num_layers=cfg["num_layers"],
                z_dim=cfg["z_dim"],
                kernel_size=cfg["kernel_size"],
                stride=cfg["stride"],
                batch_norm=cfg["batch_norm"],
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
            choices=["ConvLayers", "DeconvLayersi"])
    parser.add_argument("--test-input-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--num-max-chans", type=int, default=512)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--pg", "--progressive", dest="progressive",
            action="store_true")
    args = parser.parse_args()
    _test_main(args)

