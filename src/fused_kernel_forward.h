#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> bn_relu_forward(torch::Tensor tensor, torch::Tensor lambdas, torch::Tensor betas);

