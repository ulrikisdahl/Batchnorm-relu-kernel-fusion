add_executable(fused_ops fused_ops.cpp)

#link against Torch libraries
target_link_libraries(fused_ops PRIVATE ${TORCH_LIBRARIES})