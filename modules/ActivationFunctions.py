#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:44:32 2020

@author: briardoty
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class Swish(nn.Module):
    """
    Pytorch nn module implementation of swish activation function
    """
    
    def __init__(self, beta=1.0):
        
        super(Swish, self).__init__()
        self.beta = float(beta)
    
    def forward(self, input_tensor):
        
        return input_tensor * torch.sigmoid(self.beta * input_tensor)

class Renlu(nn.Module):
    """
    Pytorch nn module implementation of "renlu" activation function
    where renlu(x, alpha) = 0 if x <=0 else x^alpha
    """
    
    def __init__(self, alpha=0.5):
        
        super(Renlu, self).__init__()
        self.alpha = float(alpha)
    
    def forward(self, input_tensor):
        
        output = torch.relu(input_tensor)

        idxs = output.nonzero(as_tuple=True)
        output[idxs] = output[idxs].pow(self.alpha)
        
        return output
    
class SanityCheck(nn.Module):
    """
    Pytorch nn module to implement sanity check activation function
    """
    
    def __init__(self, _):
        
        super(SanityCheck, self).__init__()
        self._ = _ # throwaway param
    
    def forward(self, input_tensor):
        
        return input_tensor * 0
    
class Heaviside(nn.Module):
    """
    Pytorch nn module to implement step function for use as activation fn
    """
    
    def __init__(self, x2):
        
        super(Heaviside, self).__init__()
        self.x2 = x2
    
    def forward(self, input_tensor):
        
        output = Variable(input_tensor.new(input_tensor.size()))
        
        output[:] = np.heaviside(input_tensor.detach().cpu().numpy(), self.x2)
    
        return output
    
    
    