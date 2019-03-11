"""
General utilities for working with the code for Packt Fundamentals of GANs
"""

import logging
import numpy as np
import os
import sys
import torch

pe = os.path.exists
pj = os.path.join

def compute_features(model, data_loader, gpu_device=0, make_chip_list=True):
    logging.debug("compute_features")
    logging.info("Generating features ...")
    features = []
    chip_list = []
    if torch.cuda.is_available():
        model = model.cuda(gpu_device)
    if make_chip_list:
        data_loader.dataset.set_return_names(True)
        for inputs,_,name in data_loader:
            inputs = torch.autograd.Variable(inputs.cuda(gpu_device))
            outputs = model.get_features(inputs)
            features.append( outputs.cpu().data.numpy() )
            chip_list.append(name)
        data_loader.dataset.set_return_names(False)
        chip_list = np.squeeze( np.concatenate(chip_list) )
    else:
        for inputs,_ in data_loader:
            inputs = torch.autograd.Variable(inputs.cuda(gpu_device))
            outputs = model.get_features(inputs)
            features.append( outputs.cpu().data.numpy() )
    model = model.cpu()
    features = np.concatenate(features)
    logging.info("... Done, %d features generated." % (len(features)))
    features = np.squeeze( features )
    if len(features.shape) > 2:
        raise RuntimeError("Feature matrix has too many dimensions:",
            features.shape)
    if make_chip_list:
        return features,chip_list
    return features

def create_session_dir(sessions_dir):
    ct = 0
    while pe(pj(sessions_dir, "session_%02d" % ct)):
        ct += 1
    session_dir = pj(sessions_dir, "session_%02d" % ct)
    os.makedirs(session_dir) 
    return session_dir

def init_session_log(cfg, filemode="w"):
    logging.basicConfig(level=logging.DEBUG,
            filename=pj(cfg["session_dir"], "session.log"),
            filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    logging.info("Command line:")
    cmdln_args = " ".join(sys.argv)
    logging.info("%s\n" % cmdln_args)

    logging.info("Full parameter list")
    for k,v in cfg.items():
        logging.info("%s: %s" % (k, repr(v)))

def show_devices():
    if not torch.cuda.is_available():
        print("There are no CUDA devices available on this system.")
        return
    print("CUDA devices:")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print("\tDevice %d: %s, %s" % (i, name, capability))

def z_sampler(batch_size, z_dim, cudev):
    if cudev >= 0:
        z = torch.cuda.FloatTensor(batch_size, z_dim).normal_(0.0,1.0)
    else:
        z = torch.FloatTensor(batch_size, z_dim).normal_(0.0,1.0)
    return z


