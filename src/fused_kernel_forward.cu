#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <iostream>
#include <vector>
#include <ATen/Dispatch.h>


//__device__ - can only be called on GPU
template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t z){
    return z > 0 ? z : 0;
}

/**
 * Workaround for templated dynamic shared memory from: https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory 
 */
template <typename T>
__device__ T* shared_memory_proxy()
{
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

template <typename scalar_t> //works because ATEN abstracts away datatype 
__global__ void bn_relu_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> tensor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> lambdas,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> betas,
    int N, int C, int H, int W
    ){

    //the shared memory is split into two parts:
    // - the first part is in range [0, N) - this is where the tensor values are stored
    // - the second part is in range [N, 2N) - this is where the values used to compute the statistics are stored 
    //auto sharedMemory = shared_memory_proxy<scalar_t>();
    extern __shared__ float sharedMemory[];

    int batch_idx = threadIdx.x; //represents the element along the batch dimension we are processing
    int channel_idx = blockIdx.x;
    int height_idx = blockIdx.y;
    int width_idx = blockIdx.z;


    if (batch_idx < N && channel_idx < C && height_idx < H && width_idx < W){
        //step 0: Load into shared memory
        scalar_t thread_value = tensor[batch_idx][channel_idx][height_idx][width_idx];
        sharedMemory[threadIdx.x] = thread_value; //allocates memory addresses in range [0, N)
        sharedMemory[N + threadIdx.x] = thread_value; //NB!
        __syncthreads(); 

        //Step 1: Sum reduction
        for (int stride_div = 2; stride_div <= N; stride_div *= 2){
            if (threadIdx.x < (N / stride_div)){ //TODO: Does it matter if this is not offset by N?
                sharedMemory[N + threadIdx.x] += sharedMemory[N + threadIdx.x + (N / stride_div)];
            }
            __syncthreads();
        }

        //Step 2: Compute batch statistics per feature. In the range [N, 2N)
        scalar_t mean = sharedMemory[N] / N;
        sharedMemory[N + threadIdx.x] = (sharedMemory[threadIdx.x] - mean) * (sharedMemory[threadIdx.x] - mean); //important to not offset by N on the right
        __syncthreads();

        for (int stride_div = 2; stride_div <= N; stride_div *=2 ){
            if (threadIdx.x < (N / stride_div)){
                sharedMemory[N + threadIdx.x] += sharedMemory[N + threadIdx.x + (N / stride_div)];
            }
            __syncthreads();
        }
        
        scalar_t epsilon = 1e-5;
        scalar_t stddev = sqrt(sharedMemory[N] / N + epsilon);

        //Step 3: Batch Normalization + ReLU. In range [0, N)
        scalar_t lambda = lambdas[channel_idx][height_idx][width_idx]; //TODO: improve access 
        scalar_t beta = betas[channel_idx][height_idx][width_idx];
        sharedMemory[threadIdx.x] = relu(lambda * (sharedMemory[threadIdx.x] - mean) / stddev + beta); //unecessary write
        __syncthreads();

        //copy result over to HBM
        tensor[batch_idx][channel_idx][height_idx][width_idx] = sharedMemory[threadIdx.x]; 
    }
}

/**
 * @brief Fuses batch normalization step and ReLU activation step to one kernel
 * 
 * @param tensor: input tensor of shape (batch_size, channels, height, width)
 * @param lambdas: scaling factor for each feature, of shape (C, H, W)
 * @param beta: offsets for batch normalization, of shape (C, H, W) 
 */
std::vector<torch::Tensor> bn_relu_forward(torch::Tensor tensor, torch::Tensor lambdas, torch::Tensor betas){
    int N = tensor.sizes()[0]; //batch_size
    int C = tensor.sizes()[1];
    int H = tensor.sizes()[2];
    int W = tensor.sizes()[3];

    int BLOCK_SIZE = N;

    dim3 grid(C, H, W); //each block contains one feature (along a batch), and each grid contains all features  
    int blockDimension = BLOCK_SIZE; 
    
    AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "bn_relu_forward", ([&] { //abstracts away boilerplate needed to handle different float types
        bn_relu_forward_kernel<<<grid, blockDimension>>>(
            tensor.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            lambdas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            N, C, H, W
        );
    }));

    return {tensor};
}