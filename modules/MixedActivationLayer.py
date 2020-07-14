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
    from .ActivationFunctions import Renlu, Swish, SanityCheck
except:
    from ActivationFunctions import Renlu, Swish, SanityCheck


# map act fn names to fns themselves
act_fn_dict = {
    "relu": torch.relu,
    "renlu": Renlu,
    "swish": Swish,
    "sanityCheck": SanityCheck
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
                 verbose=False):
        
        super(MixedActivationLayer, self).__init__()
        
        self.act_fns = get_activation_fns(act_fn_names, act_fn_params)
        print("Initialized MixedActivationLayer with the following activation"
              + f"functions: {self.act_fns}")
        self.masks = generate_masks(n_features, len(self.act_fns), n_repeat)
        
        self.verbose = verbose
        
    def forward(self, input_tensor):
        
        output = Variable(input_tensor.new(input_tensor.size()))
        
        for act_fn, mask in zip(self.act_fns, self.masks):
            output[:,mask] = act_fn(input_tensor[:,mask])
            
        if self.verbose:
            print(f"Input: {input_tensor}")
            print(f"Output: {output}")
        
        return output
    
    







