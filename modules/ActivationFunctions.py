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

class Renluf(torch.autograd.Function):
    """
    Pytorch torch.autograd.Function implementation of "renlu" act fn
    with backward() implemented
    """

    # def __init(self, alpha=1.):

    #     super(Renluf, self).__init__()
    #     self.alpha = float(alpha)

    @staticmethod
    def forward(ctx, input, alpha):

        ctx.save_for_backward(input) # save input for backward pass
        ctx.alpha = alpha

        # rectify
        output = torch.relu(input)

        # apply exponent
        idxs = output.nonzero(as_tuple=True)
        output[idxs] = output[idxs].pow(alpha)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        saved_input, = ctx.saved_tensors
        grad_input = grad_alpha = None # won't actually need grad_alpha since alpha is static

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[saved_input < 0] = 0
            grad_input[saved_input > 0] = grad_input[saved_input > 0].pow(ctx.alpha)
        
        if torch.isnan(grad_input).any().item():
            torch.set_printoptions(profile="full")
            print("Saved input:")
            print(saved_input)
            print()

            print("Grad output:")
            print(grad_output)
            print()

            print("Grad input:")
            print(grad_input)
            print()
            torch.set_printoptions(profile="default")

        return grad_input, grad_alpha

        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.clone()

        #     # get lists of odd and even indices
        #     input_shape = input.shape[0]
        #     even_indices = [i for i in range(0, input_shape, 2)]
        #     odd_indices = [i for i in range(1, input_shape, 2)]

        #     # set grad_input for even_indices
        #     grad_input[even_indices] = (input[even_indices] >= 0).float() * grad_input[even_indices]

        #     # set grad_input for odd_indices
        #     grad_input[odd_indices] = (input[odd_indices] < 0).float() * grad_input[odd_indices]

        # return grad_input

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
        
        return Renluf.apply(input_tensor, self.alpha)

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
    
    
    