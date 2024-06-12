#include <iostream>
#include <torch/extension.h>
#include <pybind11/pybind11.h> 

torch::Tensor d_func(){
    return torch::rand({2, 3});
}

PYBIND11_MODULE(testcpp, handle){
    handle.def("call_python", &d_func); //first argument is the name of the function in python and second argument is a reference to the funciton we want to bind
} 




