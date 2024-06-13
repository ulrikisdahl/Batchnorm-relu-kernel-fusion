import torch
from torch.autograd import Function
from binder import bn_relu_forward, bn_relu_backward

#TODO: Create lambda and beta parameters 

class BatchNormRelu(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        """
        Args:
            ctx: context object
            inpu_tensor: input tensor to the batchnorm operation
        """

        ctx.save_for_backward(input_tensor) #stash information for backward pass
        output = bn_relu_forward(input_tensor)
        return output
    
    @staticmethod
    def backward(ctx, grad_output): #must have same numer of arguments as outputs in forward
        """
        Args:
            ctx: context object
            grad_output: gradient of the output from the forward operation
        """
        
        return 

        





