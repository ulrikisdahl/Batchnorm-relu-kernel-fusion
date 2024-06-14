#include <torch/extension.h>
#include <vector>
#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ __forceinline__ scalar_t d_relu(scalar_t z){
    return z > 0 ? 1 : 0;
}


/**
 * @brief computes the backward pass of the batch normalization and relu activation function
 * 
 * @param output_grad the gradient of the loss with respect to the output of the forward pass
 * @param input the input to the forward pass
 * @param mean the mean calculates from the forward pass
 */
template <typename scalar_t> 
__global__ void bn_relu_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output_grad,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input_grad,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> lambdas,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> betas,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> lambdas_grad,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> betas_grad,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> stddev,
    int N, int C, int H, int W
) {
    extern __shared__ float sharedMemory[]; 

    int batch_idx = threadIdx.x;
    int channel_idx = blockIdx.x;
    int height_idx = blockIdx.y;
    int width_idx = blockIdx.z;

    if (batch_idx < N && channel_idx < C && height_idx < H && width_idx < W){ 
        //recompute the normalized value
        scalar_t mean_value = mean[channel_idx][height_idx][width_idx]; //WARNING
        scalar_t stddev_value = stddev[channel_idx][height_idx][width_idx];
        scalar_t normalized_value = (input[batch_idx][channel_idx][height_idx][width_idx] - mean_value) / stddev_value;

        //compute the gradient of the input w.r.t. the loss
        scalar_t reconstructed_output = lambdas[channel_idx][height_idx][width_idx] * normalized_value + betas[channel_idx][height_idx][width_idx];
        scalar_t relu_grad = d_relu(reconstructed_output) * output_grad[batch_idx][channel_idx][height_idx][width_idx]; //the delta
        input_grad[batch_idx][channel_idx][height_idx][width_idx] = relu_grad * lambdas[channel_idx][height_idx][width_idx] / stddev_value; //dL/dx
    

        //compute gradients for lambdas and betas
        sharedMemory[threadIdx.x] = relu_grad * normalized_value; //individual gradient for lambda on one sample
        sharedMemory[N + threadIdx.x] = relu_grad; //individual gradient for beta on one sample
        __syncthreads();
        
        if (threadIdx.x == 0){ //TODO: Use sum reduction
            scalar_t grad_lambda = 0;
            scalar_t grad_beta = 0;
            for(int i = 0; i < N; i++){
                grad_lambda += sharedMemory[i]; //sum gradients
                grad_beta += sharedMemory[N + i];
            }
            atomicAdd(&lambdas_grad[channel_idx][height_idx][width_idx], grad_lambda);
            atomicAdd(&betas_grad[channel_idx][height_idx][width_idx], grad_beta);
        }
    }
}


/**
 * @brief launches the bn_relu_backward_kernel kernel
 */
std::vector<torch::Tensor> bn_relu_backward(
    torch::Tensor output_grad, 
    torch::Tensor input,
    torch::Tensor input_grad,
    torch::Tensor lambdas,
    torch::Tensor betas, 
    torch::Tensor lambdas_grad,
    torch::Tensor betas_grad,
    torch::Tensor mean,
    torch::Tensor stddev){
        
    int N = output_grad.size(0);
    int C = output_grad.size(1);
    int H = output_grad.size(2);
    int W = output_grad.size(3);

    dim3 grid(C, H, W);
    int blockDimension = N;

    AT_DISPATCH_FLOATING_TYPES(output_grad.scalar_type(), "bn_relu_backward", ([&] {
        bn_relu_backward_kernel<<<grid, blockDimension>>>(
            output_grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            input_grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            lambdas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            lambdas_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            betas_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            mean.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            stddev.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            N, C, H, W
        );
    }));

    return {input_grad, lambdas_grad, betas_grad}; //
}