#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:03:23 2020

@author: briardoty
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


def generate_masks(n_features, n_fns):
    # TODO: generate better masks? (outside this module, pass them in?)
    masks = []    
    n_period = n_features / n_fns
    
    for i in range(n_fns):
        mask = [j for j in range(n_features) if (j >= i*n_period and j < (i+1)*n_period)]
        masks.append(mask)
    
    return masks

class MixedActivationLayer(nn.Module):
    
    def __init__(self, n_features):
        
        super(MixedActivationLayer, self).__init__()
        self.n_features = n_features
        
        # TODO: pass activation functions in here
        self.activation_fns = [torch.relu, torch.tanh, torch.sigmoid]
        self.masks = generate_masks(self.n_features, len(self.activation_fns))
        
    def forward(self, input_tensor):
        output = Variable(input_tensor.new(input_tensor.size()))
        
        for act_fn, mask in zip(self.activation_fns, self.masks):
            output[:,mask] = act_fn(input_tensor[:,mask])
            
        return output
    
