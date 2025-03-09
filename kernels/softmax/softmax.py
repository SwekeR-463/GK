import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr, # lowest power-of-2 greater than n_cols
    num_stages: tl.constexpr,
    num_wraps: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # tl.range acts as an iterator
        row_start_ptr = input_ptr + row_idx * input_row_stride
        
        # load the row into the SRAM
        col_offsets = tl.arange(0, BLOCK_SIZE)
        # tl.arange provides an array of values
        
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        
        row = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        
        # subtract max for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        
        softmax_output = numerator / denominator
        
        # write the output back to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask = mask)
        
# fetching a dictionary full of the GPU's specifications
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]

TOTAL_SRAM_PER_SM = properties["max_shared_mem"]

WARP_SIZE = properties["warpSize"] # 32

# wrapper function
def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    num_wraps = 4
    if BLOCK_SIZE >= 2048:
        num_wraps = 8
    if BLOCK_SIZE >= 4096:
        num_wraps = 16
        
    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2
    
    y = torch.empty_like(x)
    
    kernel = _softmax_kernel.warmup(x, y,
                                    x.stride(0), y.stride(0),
                                    n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE,
                                    num_stages=num_stages,
                                    num_wraps=num_wraps,
                                    grid=(1,))
    # .warmup pre compiles kernel & tells us how many registers and how much shared memory it needs
    
    # info from the warmup process gave
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    
    # reg based occupancy
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_wraps)
    
    # shared memory-based occupancy
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    # determines how many programs can run per SM based on register usage and shared memory usage
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    # how many programs to run in total
    num_programs = min(NUM_SM * programs_per_sm, n_rows)
    
    # grid config
    grid = (num_programs, 1, 1)
    
    # launch the kernelll
    kernel[grid](
        x, y,
        x.stride(0), y.stride(0), 
        n_rows, n_cols,
    )
    
    return y