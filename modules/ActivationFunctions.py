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

    def __repr__(self):
        
        return f"Swish(beta={self.beta})"
    
    def forward(self, input_tensor):
        
        return input_tensor * torch.sigmoid(self.beta * input_tensor)

class HSwish(nn.Module):
    """
    Pytorch nn module implementation of hswish activation function
    """
    
    def __init__(self, beta=1.0):
        
        super(HSwish, self).__init__()
        self.beta = float(beta)

    def __repr__(self):
        
        return f"HSwish(beta={self.beta})"
    
    def forward(self, input_tensor):
        
        return input_tensor * torch.relu(input_tensor + self.beta/2.) / self.beta

class Renlu(nn.Module):
    """
    Pytorch nn module implementation of "renlu" activation function
    where renlu(x, alpha) = 0 if x <=0 else x^alpha
    """
    
    def __init__(self, alpha=0.5):
        
        super(Renlu, self).__init__()
        self.alpha = float(alpha)
    
    def __repr__(self):
        
        return f"Renlu(alpha={self.alpha})"
    
    def forward(self, input_tensor):
        
        output = torch.relu(input_tensor)

        idxs = output.nonzero(as_tuple=True)
        output[idxs] = output[idxs].pow(self.alpha)
        
        if (torch.isnan(output).any().item()):
            torch.set_printoptions(profile="full")
            print("Renlu input:")
            print(input_tensor)
            print()

            rect = torch.relu(input_tensor)
            print("Rect:")
            print(rect)
            print()

            print("Renlu output 1:")
            print(output)
            print()

            output2 = torch.relu(input_tensor)
            idxs = output2.nonzero(as_tuple=True)
            output2[idxs] = output2[idxs].pow(self.alpha)
            print("Renlu output 2:")
            print(output2)
            print()

            torch.set_printoptions(profile="default")

        return output

class Sigfreud(nn.Module):
    """
    Pytorch nn module implementation of "Sigfreud" activation function
    where sigfreud(x, beta) = sigmoid(beta * x)
    """
    
    def __init__(self, beta=1.0):
        
        super(Sigfreud, self).__init__()
        self.beta = float(beta)

    def __repr__(self):
        
        return f"Sigfreud(beta={self.beta})"
    
    def forward(self, input_tensor):
        
        return self.beta ** torch.sigmoid(input_tensor) - 1

class Tanhe(nn.Module):
    """
    Pytorch nn module implementation of "Tanhe" activation function
    where Tanhe(x, beta) = (e^*x*beta-e^-x)/(e^x*beta+e^-x)
    """
    
    def __init__(self, beta=1.0):
        
        super(Tanhe, self).__init__()
        self.beta = float(beta)

    def __repr__(self):
        
        return f"Tanhe(beta={self.beta})"
    
    def forward(self, input_tensor):
        
        # return torch.tanh(input_tensor)
        top = torch.exp(torch.mul(self.beta, input_tensor)) - torch.exp(torch.neg(input_tensor))
        bot = torch.exp(torch.mul(self.beta, input_tensor)) + torch.exp(torch.neg(input_tensor))
        
        return torch.div(top, bot)

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
    
    
    