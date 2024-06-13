#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> bn_relu_forward(torch::Tensor tensor, float lambda, float beta);

