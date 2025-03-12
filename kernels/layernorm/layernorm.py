import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr, 
    w_ptr, b_ptr, 
    mean_ptr, rstd_ptr, 
    stride_M, 
    N, 
    eps, 
    BLOCK_SIZE: tl.constexpr,
):
    # Compute mean and variance for each row
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M
    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        sum_accumulator += x_ptrs
    mean = tl.sum(sum_accumulator, axis=0) / N

    # Compute variance and reciprocal standard deviation
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        diff = tl.where(cols < N, x_ptrs - mean, 0.)
        acc += diff * diff
    var = tl.sum(acc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Normalize and apply linear transformation
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w_ptrs = tl.load(w_ptr + cols, mask=mask)
        b_ptrs = tl.load(b_ptr + cols, mask=mask)
        x_ptrs = tl.load(x_ptr + cols, mask=mask)
        x_hat = (x_ptrs - mean) * rstd
        y = x_hat * w_ptrs + b_ptrs
        tl.store(y_ptr + cols, y, mask=mask)

@triton.jit
def _layernorm_backward_dLdx(
    x_ptr, dLdx_ptr, dLdy_ptr, 
    w_ptr, 
    dLdw_intermediate_ptr, dLdb_intermediate_ptr, 
    mean_ptr, rstd_ptr, 
    locks_ptr, 
    stride, N, 
    GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Compute dLdx and accumulate partial sums for dLdw and dLdb
    PID = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += PID * stride
    dLdx_ptr += PID * stride
    dLdy_ptr += PID * stride
    x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + PID)
    rstd = tl.load(rstd_ptr + PID)
    x_normalized = tl.where(mask, (x - mean) * rstd, 0.)
    dydx_normed = tl.where(mask, w * dLdy, 0.)
    c1 = tl.sum(x_normalized * dydx_normed, axis=0) / N
    c2 = tl.sum(dydx_normed, axis=0) / N
    dLdx = (dydx_normed - (x_normalized * c1 + c2)) * rstd
    tl.store(dLdx_ptr + cols, dLdx, mask=mask)

    # Accumulate partial sums for dLdw and dLdb using locks
    dLdw_contribution = (dLdy * x_normalized).to(torch.float64)
    dLdb_contribution = (dLdy).to(torch.float64)
    lock_id = PID % GROUP_SIZE
    locks_ptr += lock_id
    count_ptr = locks_ptr + GROUP_SIZE
    dLdw_intermediate_ptrs = dLdw_intermediate_ptr + lock_id * N + cols
    dLdb_intermediate_ptrs = dLdb_intermediate_ptr + lock_id * N + cols
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
    count = tl.load(count_ptr)
    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
    else:
        dLdw_contribution += tl.load(dLdw_intermediate_ptrs, mask=mask)
        dLdb_contribution += tl.load(dLdb_intermediate_ptrs, mask=mask)
    tl.store(dLdw_intermediate_ptrs, dLdw_contribution, mask=mask)
    tl.store(dLdb_intermediate_ptrs, dLdb_contribution, mask=mask)
    tl.atomic_xchg(locks_ptr, 0)

@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_intermediate_ptr, dLdb_intermediate_ptr, 
    dLdw_ptr, dLdb_ptr, 
    GROUP_SIZE, N, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Accumulate intermediate partial sums into final dLdw and dLdb
    PID = tl.program_id(0)
    col_ptrs = PID * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs[:, None] < GROUP_SIZE) & (col_ptrs[None, :] < N)
        offsets = row_ptrs[:, None] * N + col_ptrs[None, :]
        dLdw_acc += tl.load(dLdw_intermediate_ptr + offsets, mask=mask, other=0.)
        dLdb_acc += tl.load(dLdb_intermediate_ptr + offsets, mask=mask, other=0.)
    sum_dLdw = tl.sum(dLdw_acc, axis=0)
    sum_dLdb = tl.sum(dLdb_acc, axis=0)
    tl.store(dLdw_ptr + col_ptrs, sum_dLdw, mask=col_ptrs < N)
    tl.store(dLdb_ptr + col_ptrs, sum_dLdb, mask=col_ptrs < N)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # Reshape input and allocate output tensors
        M, N = x.reshape(-1, x.shape[-1]).shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        y = torch.empty_like(x)

        # Determine block size and number of warps
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # Launch forward kernel
        _layernorm_forward[(M, )](
            x, y, weight, bias,
            mean, rstd,
            x.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )

        # Save tensors and metadata for backward pass
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dLdy):
        # Retrieve saved tensors and metadata
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        # Allocate gradient tensors
        dLdw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dLdb = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dLdx = torch.empty_like(dLdy)

        # Determine group size for parallel reduction
        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256

        # Allocate intermediate tensors and locks
        dLdw_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        dLdb_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)

        # Launch first backward kernel to compute dLdx and partial sums
        _layernorm_backward_dLdx[(M, )](
            x, dLdx, dLdy,
            w, dLdw_intermediate, dLdb_intermediate,
            mean, rstd,
            locks,
            x.stride(0), N,
            GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE_N=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
        )

        # Launch second backward kernel to accumulate partial sums
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layernorm_backward_dLdw_dLdb[grid](
            dLdw_intermediate, dLdb_intermediate, dLdw, dLdb,
            min(GROUP_SIZE, M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128,
        )

        # Return gradients for all inputs
        return dLdx, None, dLdw, dLdb, None
    
    
# Test forward and backward passes
# Test forward and backward passes
x = torch.randn(4096, device='cuda', requires_grad=True, dtype=torch.float64)  # Use float64
weight = torch.ones(4096, device='cuda', dtype=torch.float64, requires_grad=True)  # Use float64
bias = torch.zeros(4096, device='cuda', dtype=torch.float64, requires_grad=True)  # Use float64
eps = 1e-5

# Forward pass
y = LayerNorm.apply(x, None, weight, bias, eps)

# Backward pass
dy = torch.ones_like(y)  # Assume dL/dy = 1
dx = torch.autograd.grad(y, x, grad_outputs=dy)[0]

print("Forward pass output:", y)
print("Backward pass gradients (dx):", dx)

# Gradient check
test = torch.autograd.gradcheck(LayerNorm.apply, (x, None, weight, bias, eps), eps=1e-6, atol=1e-5, nondet_tol=1e-5)
print("Gradient check passed:", test)