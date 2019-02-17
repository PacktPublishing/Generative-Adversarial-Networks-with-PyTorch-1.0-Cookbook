"""
General utilities for working with the code for Packt Fundamentals of GANs
"""

import logging
import os
import sys
import torch

pe = os.path.exists
pj = os.path.join

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

