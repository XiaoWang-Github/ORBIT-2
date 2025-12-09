import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def int8_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down.
    stride_am, stride_ak,
    stride_bk, stride_bn, # Transposed B: stride_bk is row stride, stride_bn is col stride
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A has shape (M, K), int8
    B has shape (K, N), int8 (We expect B to be passed as Transposed (N, K) from python but viewed as (K, N) here? 
    Wait, standard gemm is A(M,K) * B(K,N). 
    If we pass B.t() from python which is (N, K), we need to handle strides correctly.
    Let's assume standard row-major A(M, K) and Column-major B(K, N) or similar.
    Actually, to keep it simple, we will pass A and B such that the dot product is over K.
    """
    
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptr` is initially pointing to the first element of A.
    # `b_ptr` is initially pointing to the first element of B.
    
    # Current block's row indices for A
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # Current block's col indices for B
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    # k indices for the loop
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers
    # A is (M, K). Ptr = base + row_idx * stride_am + col_idx * stride_ak
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # B is (K, N). Ptr = base + row_idx * stride_bk + col_idx * stride_bn
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        
        # Advance the pointers to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_int8_matmul(a, b):
    """
    Performs Matrix Multiplication C = A x B
    A: (M, K) int8
    B: (N, K) int8 (Transposed in Python side usually, so logically (K, N))
    Wait, let's clarify dimensions.
    Standard matmul: (M, K) @ (K, N) -> (M, N)
    
    If we pass B as (N, K) [Transposed state], we want to treat it as (K, N).
    So striding of B passed to kernel should be set to reflect (K, N) view.
    If B is (N, K) row-major:
      stride(0) = K, stride(1) = 1.
      Element at (n, k) is at offset n*K + k.
    If we want to view it as (K, N):
      Element at (k, n) should be at offset k + n*K.
      So stride_row (stride_bk) = 1
      So stride_col (stride_bn) = K
      
    HOWEVER, `PureInt8Matmul` logic passes `weight.t()` which makes it (In, Out) = (K, N).
    But `weight.t()` in PyTorch just swaps strides.
    
    Let's stick to standard interface:
    A: (M, K)
    B: (K, N)
    Output: (M, N)
    
    The user must ensure A and B are shaped/strided correctly.
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    # b can be non-contiguous (e.g. transposed), but triton load handles strides.
    # Optimally, for column-major access in B (since we iterate K along rows of B), 
    # B should be column-major (i.e. Transposed of a Row-Major matrix).
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    int8_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=None
    )
    
    return c
