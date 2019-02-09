"""
General utilities for working with the code for Packt Fundamentals of GANs
"""

import os
import sys
import torch

pe = os.path.exists
pj = os.path.join

def create_session_dir(sessions_dir):
    ct = 0
    while pe(pj(sessions_dir, "session_%02d")):
        ct += 1
    return pj(sessions_dir, "session_%02d" % (ct))

def init_session_log(cfg):
    with open(pj(cfg["session_dir"], "session.log"), "w") as fp:
        fp.write("Command line:\n")
        cmdln_args = " ".join(sys.argv)
        fp.write("%s\n\n" % cmdln_args)

        print("Full parameter list")
        for k,v in cfg.items():
            fp.write("%s: %s\n" % (k, repr(v)))

def show_devices():
    if not torch.cuda.is_available():
        print("There are no CUDA devices available on this system.")
        return
    print("CUDA devices:")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print("\tDevice %d: %s, %s" % (i, name, capability))

