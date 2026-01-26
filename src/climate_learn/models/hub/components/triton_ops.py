import torch
import triton
import triton.language as tl
import os

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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def int8_matmul_out_int8_kernel(
    a_ptr, b_ptr, c_ptr, scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    scale = tl.load(scale_ptr)
    scaled = acc.to(tl.float32) / scale
    q = tl.floor(scaled + 0.5)
    q = tl.where(q < -127, -127, q)
    q = tl.where(q > 127, 127, q)
    q = q.to(tl.int8)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, q, mask=c_mask)


@triton.jit
def int8_matmul_out_int8_kernel_fixed(
    a_ptr, b_ptr, c_ptr, scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    scale = tl.load(scale_ptr)
    scaled = acc.to(tl.float32) / scale
    q = tl.floor(scaled + 0.5)
    q = tl.where(q < -127, -127, q)
    q = tl.where(q > 127, 127, q)
    q = q.to(tl.int8)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, q, mask=c_mask)


def triton_int8_linear_out_int8(a, b, scale):
    """
    Matrix multiply with int8 output and per-tensor scale.
    a: (M, K) int8
    b: (K, N) int8
    scale: 0-dim float tensor on CUDA
    returns: (M, N) int8
    """
    assert a.is_cuda and b.is_cuda, "triton_int8_linear_out_int8 expects CUDA tensors"
    assert a.dtype == torch.int8 and b.dtype == torch.int8, "a/b must be int8"
    assert a.dim() == 2 and b.dim() == 2, "a/b must be 2D"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert scale.is_cuda, "scale must be on CUDA"
    if scale.numel() != 1:
        raise ValueError("scale must be a single-element tensor")

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int8)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    int8_matmul_out_int8_kernel[grid](
        a, b, c, scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def triton_int8_linear_out_int8_fixed(
    a,
    b,
    scale,
    block_m,
    block_n,
    block_k,
    group_m,
    num_warps,
    num_stages,
):
    """
    Matrix multiply with int8 output using fixed tiling.
    """
    assert a.is_cuda and b.is_cuda, "triton_int8_linear_out_int8_fixed expects CUDA tensors"
    assert a.dtype == torch.int8 and b.dtype == torch.int8, "a/b must be int8"
    assert a.dim() == 2 and b.dim() == 2, "a/b must be 2D"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert scale.is_cuda, "scale must be on CUDA"
    if scale.numel() != 1:
        raise ValueError("scale must be a single-element tensor")
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block sizes must be positive")

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int8)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    int8_matmul_out_int8_kernel_fixed[grid](
        a, b, c, scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def int8_matmul_out_int8_varattn_kernel(
    a_ptr, b_ptr, c_ptr, scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    scale = tl.load(scale_ptr)
    inv_scale = 1.0 / scale
    scaled = acc.to(tl.float32) * inv_scale
    q = tl.floor(scaled + 0.5)
    q = tl.where(q < -127, -127, q)
    q = tl.where(q > 127, 127, q)
    q = q.to(tl.int8)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, q, mask=c_mask)


@triton.jit
def int8_matmul_out_int8_varattn_kernel_fixed(
    a_ptr, b_ptr, c_ptr, scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    scale = tl.load(scale_ptr)
    inv_scale = 1.0 / scale
    scaled = acc.to(tl.float32) * inv_scale
    q = tl.floor(scaled + 0.5)
    q = tl.where(q < -127, -127, q)
    q = tl.where(q > 127, 127, q)
    q = q.to(tl.int8)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, q, mask=c_mask)


def _get_int_env(name, default):
    value = os.environ.get(name, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def triton_int8_linear_out_int8_varattn(a, b, scale):
    """
    VarAttention KV path: int8 matmul with int8 output using larger tiles.
    a: (M, K) int8
    b: (K, N) int8
    scale: 0-dim float tensor on CUDA
    returns: (M, N) int8
    """
    assert a.is_cuda and b.is_cuda, "triton_int8_linear_out_int8_varattn expects CUDA tensors"
    assert a.dtype == torch.int8 and b.dtype == torch.int8, "a/b must be int8"
    assert a.dim() == 2 and b.dim() == 2, "a/b must be 2D"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert scale.is_cuda, "scale must be on CUDA"
    if scale.numel() != 1:
        raise ValueError("scale must be a single-element tensor")

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int8)
    fixed_m = _get_int_env("INT8_VARATTN_KV_BLOCK_M", 0)
    fixed_n = _get_int_env("INT8_VARATTN_KV_BLOCK_N", 0)
    fixed_k = _get_int_env("INT8_VARATTN_KV_BLOCK_K", 0)
    fixed_group = _get_int_env("INT8_VARATTN_KV_GROUP_M", 8)
    fixed_warps = _get_int_env("INT8_VARATTN_KV_NUM_WARPS", 4)
    fixed_stages = _get_int_env("INT8_VARATTN_KV_NUM_STAGES", 3)
    use_fixed = fixed_m > 0 and fixed_n > 0 and fixed_k > 0
    if use_fixed:
        grid = (triton.cdiv(M, fixed_m) * triton.cdiv(N, fixed_n),)
        int8_matmul_out_int8_varattn_kernel_fixed[grid](
            a, b, c, scale,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=fixed_m,
            BLOCK_SIZE_N=fixed_n,
            BLOCK_SIZE_K=fixed_k,
            GROUP_SIZE_M=fixed_group,
            num_warps=fixed_warps,
            num_stages=fixed_stages,
        )
    else:
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        int8_matmul_out_int8_varattn_kernel[grid](
            a, b, c, scale,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['NQ', 'NK', 'K'],
)
@triton.jit
def int8_qk_bmm_kernel(
    q_ptr, k_ptr, out_ptr,
    stride_qb, stride_qm, stride_qk,
    stride_kb, stride_kn, stride_kk,
    stride_ob, stride_om, stride_on,
    NQ, NK, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_mn = tl.program_id(axis=1)
    grid_n = tl.cdiv(NK, BLOCK_N)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_ptrs = q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = k_ptr + pid_b * stride_kb + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k + k * BLOCK_K < K
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < NQ) & k_mask[None, :],
            other=0,
        )
        b = tl.load(
            k_ptrs,
            mask=(offs_n[None, :] < NK) & k_mask[:, None],
            other=0,
        )
        acc += tl.dot(q, b)
        q_ptrs += BLOCK_K * stride_qk
        k_ptrs += BLOCK_K * stride_kk

    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < NQ) & (offs_n[None, :] < NK)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_int8_bmm_qk(q, k):
    """
    Batched int8 QK^T using Triton.
    q: (B, H, NQ, K) int8
    k: (B, H, NK, K) int8
    returns: (B, H, NQ, NK) int32
    """
    assert q.is_cuda and k.is_cuda, "triton_int8_bmm_qk expects CUDA tensors"
    assert q.dtype == torch.int8 and k.dtype == torch.int8, "q/k must be int8"
    assert q.dim() == 4 and k.dim() == 4, "q/k must be 4D"
    assert q.shape[0] == k.shape[0] and q.shape[1] == k.shape[1], "batch/head mismatch"
    assert q.shape[3] == k.shape[3], "K dim mismatch"

    bsz, nheads, n_q, kdim = q.shape
    n_k = k.shape[2]
    q_3d = q.reshape(bsz * nheads, n_q, kdim).contiguous()
    k_3d = k.reshape(bsz * nheads, n_k, kdim).contiguous()

    out = torch.empty((bsz * nheads, n_q, n_k), device=q.device, dtype=torch.int32)
    grid = lambda META: (
        bsz * nheads,
        triton.cdiv(n_q, META['BLOCK_M']) * triton.cdiv(n_k, META['BLOCK_N']),
    )
    int8_qk_bmm_kernel[grid](
        q_3d, k_3d, out,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        n_q, n_k, kdim,
    )
    return out.reshape(bsz, nheads, n_q, n_k)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['NQ', 'D', 'NK'],
)
@triton.jit
def int8_av_bmm_kernel(
    a_ptr, v_ptr, out_ptr,
    stride_ab, stride_am, stride_ak,
    stride_vb, stride_vk, stride_vn,
    stride_ob, stride_om, stride_on,
    NQ, D, NK,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_mn = tl.program_id(axis=1)
    grid_n = tl.cdiv(D, BLOCK_N)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    v_ptrs = v_ptr + pid_b * stride_vb + offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(NK, BLOCK_K)):
        k_mask = offs_k + k * BLOCK_K < NK
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < NQ) & k_mask[None, :],
            other=0,
        )
        v = tl.load(
            v_ptrs,
            mask=(offs_n[None, :] < D) & k_mask[:, None],
            other=0,
        )
        acc += tl.dot(a, v)
        a_ptrs += BLOCK_K * stride_ak
        v_ptrs += BLOCK_K * stride_vk

    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < NQ) & (offs_n[None, :] < D)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_int8_bmm_av(attn, v):
    """
    Batched int8 (attn @ V) using Triton.
    attn: (B, H, NQ, NK) int8
    v: (B, H, NK, D) int8
    returns: (B, H, NQ, D) int32
    """
    assert attn.is_cuda and v.is_cuda, "triton_int8_bmm_av expects CUDA tensors"
    assert attn.dtype == torch.int8 and v.dtype == torch.int8, "attn/v must be int8"
    assert attn.dim() == 4 and v.dim() == 4, "attn/v must be 4D"
    assert attn.shape[0] == v.shape[0] and attn.shape[1] == v.shape[1], "batch/head mismatch"
    assert attn.shape[3] == v.shape[2], "NK dim mismatch"

    bsz, nheads, n_q, n_k = attn.shape
    d = v.shape[3]
    a3 = attn.reshape(bsz * nheads, n_q, n_k).contiguous()
    v3 = v.reshape(bsz * nheads, n_k, d).contiguous()

    out = torch.empty((bsz * nheads, n_q, d), device=attn.device, dtype=torch.int32)
    grid = lambda META: (
        bsz * nheads,
        triton.cdiv(n_q, META['BLOCK_M']) * triton.cdiv(d, META['BLOCK_N']),
    )
    int8_av_bmm_kernel[grid](
        a3, v3, out,
        a3.stride(0), a3.stride(1), a3.stride(2),
        v3.stride(0), v3.stride(1), v3.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        n_q, d, n_k,
    )
    return out.reshape(bsz, nheads, n_q, d)


@triton.jit
def lut_softmax_kernel(
    x_ptr,
    y_ptr,
    lut_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    n_cols,
    LUT_RANGE: tl.constexpr,
    LUT_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.maximum(x, -LUT_RANGE)
    x = tl.minimum(x, 0.0)

    step = LUT_RANGE / (LUT_SIZE - 1)
    idx = tl.floor((-x / step) + 0.5).to(tl.int32)
    idx = tl.where(idx < 0, 0, idx)
    idx = tl.where(idx > (LUT_SIZE - 1), LUT_SIZE - 1, idx)

    exp_int = tl.load(lut_ptr + idx, mask=mask, other=0).to(tl.float32)
    sum_int = tl.sum(exp_int, axis=0)
    sum_int = tl.maximum(sum_int, 1.0)
    probs = exp_int / sum_int

    tl.store(y_ptr + row * stride_ym + offs * stride_yn, probs, mask=mask)


@triton.jit
def lut_softmax_int8_kernel(
    x_ptr,
    y_ptr,
    lut_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    n_cols,
    LUT_RANGE: tl.constexpr,
    LUT_SIZE: tl.constexpr,
    OUT_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.maximum(x, -LUT_RANGE)
    x = tl.minimum(x, 0.0)

    step = LUT_RANGE / (LUT_SIZE - 1)
    idx = tl.floor((-x / step) + 0.5).to(tl.int32)
    idx = tl.where(idx < 0, 0, idx)
    idx = tl.where(idx > (LUT_SIZE - 1), LUT_SIZE - 1, idx)

    exp_int = tl.load(lut_ptr + idx, mask=mask, other=0).to(tl.float32)
    sum_int = tl.sum(exp_int, axis=0)
    sum_int = tl.maximum(sum_int, 1.0)
    probs = exp_int / sum_int

    q = tl.floor((probs * OUT_SCALE) + 0.5)
    q = tl.where(q < 0, 0, q)
    q = tl.where(q > 127, 127, q)
    q = q.to(tl.int8)

    tl.store(y_ptr + row * stride_ym + offs * stride_yn, q, mask=mask)


def triton_lut_softmax(x, lut, lut_range: float, lut_size: int):
    """
    LUT-approximated softmax over the last dimension.
    x: (M, N) tensor, float/bfloat16/float16 on CUDA.
    lut: (lut_size,) int32 tensor on CUDA.
    """
    assert x.is_cuda, "triton_lut_softmax expects CUDA tensor"
    assert x.dim() == 2, "triton_lut_softmax expects 2D tensor"
    assert lut.is_cuda, "LUT must be on CUDA"
    M, N = x.shape
    block = triton.next_power_of_2(N)
    if block > 2048:
        raise ValueError(f"Unsupported N={N} for lut softmax block={block}")

    y = torch.empty_like(x, dtype=torch.float32)
    grid = (M,)
    lut_softmax_kernel[grid](
        x, y, lut,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N,
        LUT_RANGE=lut_range,
        LUT_SIZE=lut_size,
        BLOCK_SIZE=block,
    )
    return y.to(x.dtype)


def triton_lut_softmax_int8(x, lut, lut_range: float, lut_size: int, out_scale: float):
    """
    LUT-approximated softmax over the last dimension with int8 output.
    x: (M, N) tensor, float/bfloat16/float16 on CUDA.
    lut: (lut_size,) int32 tensor on CUDA.
    out_scale: float scale factor applied before int8 quantization.
    """
    assert x.is_cuda, "triton_lut_softmax_int8 expects CUDA tensor"
    assert x.dim() == 2, "triton_lut_softmax_int8 expects 2D tensor"
    assert lut.is_cuda, "LUT must be on CUDA"
    M, N = x.shape
    block = triton.next_power_of_2(N)
    if block > 2048:
        raise ValueError(f"Unsupported N={N} for lut softmax block={block}")

    y = torch.empty_like(x, dtype=torch.int8)
    grid = (M,)
    lut_softmax_int8_kernel[grid](
        x, y, lut,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N,
        LUT_RANGE=lut_range,
        LUT_SIZE=lut_size,
        OUT_SCALE=out_scale,
        BLOCK_SIZE=block,
    )
    return y
