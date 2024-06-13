#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include "fused_ops_kernel.h"

PYBIND11_MODULE(binder, handle){
    handle.def("bn_relu_forward", &bn_relu_forward);
}

