#pragma once

#include <torch/extension.h>

std::vector<torch::Tensor> bn_relu_backward(
    torch::Tensor output_grad, 
    torch::Tensor input,
    torch::Tensor input_grad,
    torch::Tensor lambdas,
    torch::Tensor betas, 
    torch::Tensor lambdas_grad,
    torch::Tensor betas_grad, 
    torch::Tensor mean, 
    torch::Tensor stddev);