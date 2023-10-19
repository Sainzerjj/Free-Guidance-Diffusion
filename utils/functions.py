from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel

def add_dims_right(x,y):
    dim = y.ndim - x.ndim
    return x[(...,) + (None,)*dim]

def add_dims_left(x, y):
    dim = y.ndim - x.ndim
    return x[(None,)*dim + (...,)]

def normalize(x): return (x - x.min()) / (x.max() - x.min())

class UNetDistributedDataParallel(DataParallel):
    def forward(self, *inputs, **kwargs):
        # This is because the timestep (inputs[1]) in UNet is a 0-d tensor and scatter will try to split inputs[1]. We simply convert it to a float so that scatter has no effect on it.
        
        inputs = inputs[0], inputs[1].item()
        return super().forward(*inputs, **kwargs)