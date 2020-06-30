#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:03:23 2020

@author: briardoty
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
import os
from torchvision import datasets, models, transforms

def generate_masks(n_features, n_fns):
    # TODO: generate better masks? (outside this module, pass them in?)
    masks = []
    
    n_period = n_features / n_fns
    
    for i in range(n_fns):
        mask = [1 if (j >= i*n_period and j < (i+1)*n_period) else 0 for j in range(n_features)]
        masks.append(mask)
    
    return masks

class MixedActivationLayer(nn.Module):
    
    def __init__(self, input_features, output_features):
        
        # TODO: input and output are same n?
        
        super(MixedActivationLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        # TODO: pass activation functions in here
        self.activation_fns = [F.relu, F.tanh, F.sigmoid]
        self.masks = generate_masks(self.input_features, len(self.activation_fns))
        
    def forward(self, input):
        output = Variable(input.new(input.size()))
        
        for act_fn, mask in zip(self.activation_fns, self.masks):
            output[mask] = act_fn(input[mask])
            
        return output
    
