"""
General utilities for working with the code for Packt Fundamentals of GANs
"""

import torch

def show_devices():
    if not torch.cuda.is_available():
        print("There are no CUDA devices available on this system.")
        return
    print("CUDA devices:")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print("\tDevice %s: %s" % (name, capability))

