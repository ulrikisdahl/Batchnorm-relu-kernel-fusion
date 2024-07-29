import torch
import torch.nn as nn
from torch.autograd import Function
from binder import bn_relu_forward, bn_relu_backward

device="cuda"

class BatchNormReluFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, lambdas, betas):
        """
        Args:
            ctx: context object
            inpu_tensor: input tensor to the batchnorm operation
            lambdas: scaling parameter for the batchnorm operation
            betas: shift parameter for the batchnorm operation
        """

        output, means, stddevs = bn_relu_forward(input_tensor, lambdas, betas)
        ctx.save_for_backward(input_tensor, lambdas, betas, means, stddevs) #stash information for backward pass
        return output
    
    @staticmethod
    def backward(ctx, output_grad): #must have same numer of arguments as outputs in forward
        """
        Args:
            ctx: context object
            grad_output: gradient of the output from the forward operation
        """
        input_tensor, lambdas, betas, means, stddevs = ctx.saved_tensors
        input_grad = torch.empty_like(output_grad, device=output_grad.device)
        lambdas_grad = torch.zeros_like(lambdas, device=output_grad.device)
        betas_grad = torch.zeros_like(betas, device=output_grad.device) 
        input_grad, lambdas_grad, betas_grad = bn_relu_backward(
            output_grad, input_tensor, input_grad, lambdas, betas, lambdas_grad, betas_grad, means, stddevs
            )
        return input_grad, lambdas_grad, betas_grad
    
class BatchNormRelu(nn.Module):
    def __init__(self, channels, resolution):
        super(BatchNormRelu, self).__init__()

        self.lambdas = nn.Parameter(
            torch.ones(channels, resolution, resolution, device=device)
        )
        self.betas = nn.Parameter(
            torch.zeros(channels, resolution, resolution, device=device)
        )
    
    def forward(self, inp):
        return BatchNormReluFunction.apply(inp, self.lambdas, self.betas)
        


