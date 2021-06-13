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
        return rf"Swish($\beta$={self.beta}, x)"
    
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
            grad_input[saved_input <= 0] = 0
            idxs = saved_input.nonzero(as_tuple=True)
            # grad_input[idxs] = ctx.alpha * saved_input[idxs].pow(ctx.alpha - 1)
            grad_input[saved_input > 0] = ctx.alpha * saved_input[saved_input > 0].pow(ctx.alpha - 1)
        
        if (torch.isnan(grad_input).any().item() or
            not torch.isfinite(grad_input).all().item()):
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

class Relu(nn.Module):
    
    def __init__(self):
        
        super(Relu, self).__init__()
    
    def __repr__(self):
        
        return f"Relu"
    
    def forward(self, input_tensor):
        
        return torch.relu(input_tensor)

class PTanh(nn.Module):
    """
    Pytorch nn module implementation of "PTanh" activation function
    where PTanh(x, beta) = 0.5 * (tanh(x) + tanh(b*x))
    """
    
    def __init__(self, beta=1.0):
        
        super(PTanh, self).__init__()
        self.beta = float(beta)

    def __repr__(self):
        
        return rf"PTanh($\alpha$={self.beta}, x)"
    
    def forward(self, input_tensor):

        return 0.5 * torch.add(
            torch.tanh(input_tensor), 
            torch.tanh(torch.mul(self.beta, input_tensor)))

class Gompertz(nn.Module):
    """
    gompertz(x) = a*e^(-b*e^(-cx))
    """

    def __init__(self, c=1.0):
        
        super(Gompertz, self).__init__()
        self.a = 2
        self.b = float(c)
        self.c = float(c)
        self.y0 = -1

    def __repr__(self):
        
        return f"Gompertz(c={self.c})"
    
    def forward(self, input_tensor):
        
        p1 = torch.clamp(torch.mul(-self.c, input_tensor), max=50)
        exp1 = torch.exp(p1)
        p2 = torch.clamp(torch.mul(-self.b, exp1), max=50)
        exp2 = torch.exp(p2)
        
        return torch.mul(self.a, exp2)
    
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
