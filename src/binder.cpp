#include <torch/torch.h>
#include <pybind11/pybind11.h>

torch::Tensor return_tensor(){
    return torch::rand({2, 3});
}

PYBIND11_MODULE(binder, handle){
    handle.def("tensor_func", &return_tensor);
}

