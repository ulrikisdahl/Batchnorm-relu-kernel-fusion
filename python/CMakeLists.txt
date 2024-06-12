pybind11_add_module(testcpp testcpp.cpp)

target_link_libraries(testcpp PRIVATE ${TORCH_LIBRARIES}) #link pytorch agaist the module

#specify where compiler needs to look for header files
target_include_directories(testcpp PRIVATE ${TORCH_INCLUDE_DIRS})