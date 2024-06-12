cmake_minimum_required(VERSION 3.10)
project(residual_fusion)


set(CMAKE_PREFIX_PATH "/home/ulrik/libtorch")
set(CMAKE_PREFIX_PATH "/home/ulrik/.local/lib/python3.8/site-packages")

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

add_library(binder MODULE src/binder.cpp)
target_link_libraries(binder PRIVATE ${TORCH_LIBRARIES} pybind11::module)
set_target_properties(binder PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")