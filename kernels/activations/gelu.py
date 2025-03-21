import triton
import triton.language as tl
import torch

@triton.jit
def gelu_forward_kernel(
    x_ptr,  
    y_ptr,  
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr  
):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # GELU approximation using sigmoid
    gelu = x * 0.5 * (1.0 + tl.sigmoid(1.702 * x))

    tl.store(y_ptr + offsets, gelu, mask=mask)

@triton.jit
def gelu_backward_kernel(
    dy_ptr,  # pointer to gradient of loss w.r.t. output
    x_ptr,   # pointer to input tensor
    dx_ptr,  # pointer to gradient of loss w.r.t. input
    n_elements: tl.constexpr,  
    BLOCK_SIZE: tl.constexpr  
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

    # compute sigmoid-based GELU derivative
    sig = tl.sigmoid(1.702 * x)
    gelu_grad = 0.5 * (1.0 + sig + x * 1.702 * sig * (1 - sig))

    # compute dL/dx 
    dx = dy * gelu_grad

    # store gradient
    tl.store(dx_ptr + offsets, dx, mask=mask)

def gelu_forward(x):
    n_elements = x.numel()
    y = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, 1024),)  # Define number of blocks
    gelu_forward_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024, num_warps=4)

    return y

def gelu_backward(dy, x):
    n_elements = x.numel()
    dx = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, 1024),)  # Define number of blocks
    gelu_backward_kernel[grid](dy, x, dx, n_elements, BLOCK_SIZE=1024, num_warps=4)

    return dx

class GeluTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = gelu_forward(x)
        ctx.save_for_backward(x)  # save input for backward pass
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = gelu_backward(dy, x)
        return dx

# pytorch Module for GELU
class TritonGeluLayer(torch.nn.Module):
    def forward(self, x):
        return GeluTriton.apply(x)

x = torch.randn(4096, device='cuda', requires_grad=True)
y = GeluTriton.apply(x)

# backward test
dy = torch.ones_like(y)  # Assume dL/dy = 1
dx = torch.autograd.grad(y, x, grad_outputs=dy)[0]

print(y)  # forward pass output
print(dx) # backward pass gradients

from torch.autograd import gradcheck

x = torch.randn(4096, device='cuda', requires_grad=True, dtype=torch.float32)

# perform gradient check
test = gradcheck(GeluTriton.apply, (x,), eps=1e-3, atol=1e-4)
print("Gradient check passed:", test)