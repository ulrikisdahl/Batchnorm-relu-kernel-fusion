cmake_minimum_required(VERSION 3.1)
project(resnet VERSION 1.0)

#Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


#Set the path for libtorch and pybind11
set(CMAKE_PREFIX_PATH "/home/ulrik/libtorch") #libtorch 
set(CMAKE_PREFIX_PATH "/home/ulrik/.local/lib/python3.8/site-packages/pybind11") #pybind11 

#Find the Torch package
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

#Add subdirectory for the source files
add_subdirectory(src)
add_subdirectory(python)