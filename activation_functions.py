#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:44:32 2020

@author: briardoty
"""
import torch
import torch.nn as nn

class Swish(nn.Module):
    
    def __init__(self, beta=1.0):
        
        super(Swish, self).__init__()
        self.beta = beta
    
    def forward(self, input_tensor):
        
        return input_tensor * torch.sigmoid(input_tensor)

class Renlu(nn.Module):
    
    def __init__(self, alpha=0.5):
        
        super(Renlu, self).__init__()
        self.alpha = alpha
    
    def forward(self, input_tensor):
        
        return torch.relu(input_tensor) ** self.alpha