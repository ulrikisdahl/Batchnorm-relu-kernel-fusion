# Batchnorm ReLU Kernel Fusion

Batch normalization and ReLU operations fused into one kernel with backprop implementation.

</br>

## Explanation of the Backward Pass

The forward kernel can be broken up into five steps:

1. **Compute the mean**:

   $$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$

2. **Compute the variance**:

   $$\sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2$$

3. **Normalize the input**:

    $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$


4. **Scale and shift with learnable γ and β parameters**:

    $$y_i = \gamma \hat{x}_i + \beta$$

   where γ and β are learnable parameters that scale and shift the normalized value.

5. **Apply ReLU activation function**:

    $$a_i = \text{ReLU}(y_i)$$

</br>
</br>

The backward pass needs to compute both the gradients for the two sets of learnable parameters γ and β as well as the gradient of the input to the layer, which needs to be passed on to the previous layer. 


1. **Compute gradients for the learnable γ parameters**:

    $$\frac{dL}{d\gamma} = \sum_{i=1}^N \frac{dL}{da_i} \cdot \frac{da_i}{dy_i} \cdot \frac{dy_i}{d\gamma}$$


2. **Compute gradients for the learnable β parameters**:

    $$\frac{dL}{d\beta} = \sum_{i=1}^N \frac{dL}{da_i} \cdot \frac{da_i}{dy_i} \cdot \frac{dy_i}{d\beta}$$


3. **Compute the gradient of the input w.r.t. the loss**:

    $$\frac{dL}{dx_i} = \frac{dL}{da_i} \cdot \frac{da_i}{dy_i} \cdot \frac{dy_i}{dx_{\hat{i}}} \cdot \frac{d{\hat{x_i}}}{dx_i}$$

where the final term can be written as:

$$ \frac{d{\hat{x_i}}}{dx_i} = \frac{d}{dx_i} \left( \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) = \frac{1}{\sigma}$$


</br>

The implementation can be found in [fused_kernel_backward.cu](src/fused_kernel_backward.cu) 

