#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:03:23 2020

@author: briardoty
"""
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
try:
    from .ActivationFunctions import (Renlu, Swish, SanityCheck, 
    Heaviside, Sigfreud, HSwish, Tanh)
except:
    from ActivationFunctions import (Renlu, Swish, SanityCheck, 
    Heaviside, Sigfreud, HSwish, Tanh)


# map act fn names to fns themselves
act_fn_dict = {
    "relu": torch.relu,
    "torch.tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "sigfreud": Sigfreud,
    "tanh": Tanh,
    "renlu": Renlu,
    "swish": Swish,
    "sanityCheck": SanityCheck,
    "heaviside": Heaviside
}

def generate_masks(n_features, n_fns, n_repeat):
    """
    Generates a set of "masks" to apply each activation function on.

    Args:
        n_features (int): Number of input features.
        n_fns (int): Number of activation fns.
        n_repeat (int): Number of times to repeat each activation fn across layer.

    Returns:
        masks (list[list[int]]): List of "masks" to be applied for each
        activation function.

    """
    masks = [[] for _ in range(n_fns)]
    
    for i in range(n_features):
        # determine which mask this feature belongs to
        mask_idx = math.floor(i / n_repeat) % n_fns
        
        # add this feature to that mask
        masks[mask_idx].append(i)
    
    return masks

def get_activation_fns(act_fn_names, act_fn_params):
        """
        Builds list of torch activation functions for use in this layer.

        Args:
            act_fn_names (list): List of act fn names.
            act_fn_params (list): List of act fn params.

        Returns:
            act_fns (list): List of torch act fns.

        """
        act_fns = []
        
        for n, p in zip(act_fn_names, act_fn_params):
            if p is None or p == "None":
                act_fns.append(act_fn_dict[n])
            else:
                act_fns.append(act_fn_dict[n](p))
            
        return act_fns

class MixedActivationLayer(nn.Module):
    
    def __init__(self, n_features, n_repeat, act_fn_names, act_fn_params, 
        spatial=False):
        
        super(MixedActivationLayer, self).__init__()
        
        self.n_features = n_features
        self.act_fns = get_activation_fns(act_fn_names, act_fn_params)
        self.masks = generate_masks(n_features, len(self.act_fns), n_repeat)
        
        self.spatial = spatial
        if self.spatial:
            print("Activation functions will be mixed spatially!")
        
    def __repr__(self):
        return f"MixedActivationLayer(n_features={self.n_features}, act_fns={self.act_fns})"

    def forward(self, input_tensor):
        
        output = Variable(input_tensor.new(input_tensor.size()))
        
        if self.spatial:
            # apply mixing to feature map (spatial dimensions)
            output = tile(input_tensor, output, self.act_fns)
            
        else:
            # apply mixing at conv filter/unit level
            for act_fn, mask in zip(self.act_fns, self.masks):
                output[:,mask] = act_fn(input_tensor[:,mask])
        
        return output

def tile(input_tensor, output, act_fns):

    n_fns = len(act_fns)

    for i in range(n_fns):
        act_fn = act_fns[i]
        for j in range(n_fns):
            output[:, :, j::n_fns, ((i+j)%n_fns)::n_fns] = act_fn(input_tensor[:, :, j::n_fns, ((i+j)%n_fns)::n_fns])

    return output

def tile2(m, n):

    x = np.zeros((m, m), dtype=int)
    x[:,:] = -1

    for i in range(n):
        for j in range(n):
            x[j::n, ((i+j)%n)::n] = i
    
    return x
    
if __name__=="__main__":
    import numpy as np

    # first
    # x[::n, ::n] = 0
    # x[1::n, 1::n] = 0
    # x[2::n, 2::n] = 0

    # # second
    # x[::n, 1::n] = 1
    # x[1::n, 2::n] = 1
    # x[2::n, 0::n] = 1

    # # third
    # x[::n, 2::n] = 2
    # x[1::n, 0::n] = 2
    # x[2::n, 1::n] = 2

    # for i in range(n):
    #     for j in range(n):
    #         x[j::n, ((i+j)%n)::n] = i
            # x[1::n, ((i+1)%n)::n] = i
            # x[2::n, ((i+2)%n)::n] = i

    print(tile2(11,1))




