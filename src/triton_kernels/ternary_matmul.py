"""
Ternary Matmul Triton Kernel
=============================

Custom Triton kernel that exploits the Pruned Ternary weight format {-w_n, 0, +w_p}
to achieve faster inference through:

1. **2-bit weight packing**: Each ternary weight encoded as 2 bits (00=zero, 01=positive,
   10=negative). 16 weights per int32 -> 16x memory reduction vs FP32.

2. **On-the-fly decode**: Packed codes are decoded to weight values in GPU registers.
   Just bit shifts, comparisons, and conditional moves — very cheap ALU ops.

3. **Reduced memory bandwidth**: The kernel loads 16x less weight data from global memory.
   For memory-bound operations (small-batch inference), this directly translates to speedup.

The kernel computes: Y = X @ W^T where W is ternary {-w_n, 0, +w_p}.
For convolutions, use im2col (F.unfold) + this matmul kernel.
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import math


# ============================================================
# Weight Packing / Unpacking Utilities
# ============================================================

def pack_ternary_weights(W: torch.Tensor, w_p: float, w_n: float) -> torch.Tensor:
    """
    Pack FP32 ternary weights {-w_n, 0, +w_p} into 2-bit codes.

    Encoding:
      00 (0) = zero (pruned)
      01 (1) = positive (+w_p)
      10 (2) = negative (-w_n)

    16 codes packed per int32: bits [2i+1:2i] = code for position i.

    Args:
        W: (N, K) float32 tensor with values in {-w_n, 0, +w_p}
        w_p: positive scaling factor (float > 0)
        w_n: negative scaling factor (float > 0)

    Returns:
        packed: (N, ceil(K/16)) int32 tensor, contiguous
    """
    N, K = W.shape
    device = W.device
    K_padded = math.ceil(K / 16) * 16

    # Pad K to multiple of 16
    if K < K_padded:
        W_pad = torch.zeros(N, K_padded, dtype=W.dtype, device=device)
        W_pad[:, :K] = W
        W = W_pad

    # Classify: use threshold at half the scaling factor to handle FP noise
    codes = torch.zeros(N, K_padded, dtype=torch.int32, device=device)
    codes[W > w_p * 0.5] = 1   # positive
    codes[W < -w_n * 0.5] = 2  # negative

    # Pack 16 codes per int32
    codes = codes.reshape(N, -1, 16)  # (N, K_packed, 16)
    shifts = torch.arange(16, device=device, dtype=torch.int32) * 2
    packed = (codes << shifts).sum(dim=-1).to(torch.int32)

    return packed.contiguous()


def unpack_ternary_weights(packed: torch.Tensor, w_p: float, w_n: float, K: int) -> torch.Tensor:
    """Unpack 2-bit packed codes back to FP32 ternary weights. For verification."""
    N, K_packed = packed.shape
    shifts = torch.arange(16, device=packed.device, dtype=torch.int32) * 2
    codes = ((packed.unsqueeze(-1) >> shifts) & 0x3).reshape(N, -1)[:, :K]

    W = torch.zeros(N, K, dtype=torch.float32, device=packed.device)
    W[codes == 1] = w_p
    W[codes == 2] = -w_n
    return W


def weight_memory_bytes(W_packed: torch.Tensor) -> int:
    """Memory footprint of packed ternary weights in bytes."""
    return W_packed.numel() * W_packed.element_size()


def fp32_memory_bytes(N: int, K: int) -> int:
    """Memory footprint of FP32 weight matrix in bytes."""
    return N * K * 4


# ============================================================
# Triton Kernel: Ternary Matmul (Decode Approach)
# ============================================================
# Loads 2-bit packed weights, decodes to {-w_n, 0, +w_p} in registers,
# then uses tl.dot for the actual matrix multiply.
# The speedup comes from loading 16x less weight data from global memory.

_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['M', 'N', 'K_packed'])
@triton.jit
def _ternary_matmul_decode_kernel(
    # Pointers
    x_ptr, w_ptr, y_ptr, wp_ptr, wn_ptr,
    # Dimensions
    M, N, K, K_packed,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    # Block sizes (constexpr for compilation)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Ternary matmul: Y = X @ decode(W_packed)^T

    The kernel:
    1. Loads 1 packed int32 per weight row (= 16 ternary codes) per iteration
    2. Unpacks codes via bit shifts: code_i = (packed >> (2*i)) & 3
    3. Decodes to weight values: 0->0.0, 1->+w_p, 2->-w_n
    4. Uses tl.dot for the (BLOCK_M, 16) @ (16, BLOCK_N) matmul

    The ternary-specific advantage: weight data is 16x smaller than FP32,
    so global memory bandwidth for weights is reduced by 16x. For memory-bound
    operations (small-batch inference), this gives proportional speedup.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load per-layer scaling factors (scalar)
    w_p = tl.load(wp_ptr)
    w_n = tl.load(wn_ptr)
    neg_wn = -w_n

    # Bit positions for unpacking: [0, 2, 4, ..., 30]
    shifts = (tl.arange(0, 16) * 2).to(tl.int32)

    for ki in range(K_packed):
        k0 = ki * 16

        # Load packed codes: one int32 per weight row -> (BLOCK_N,)
        packed = tl.load(
            w_ptr + offs_n * stride_wn + ki,
            mask=offs_n < N, other=0
        )

        # Unpack: broadcast (BLOCK_N,) x (16,) -> (16, BLOCK_N) codes
        codes = (packed[None, :] >> shifts[:, None]) & 0x3

        # Decode to weight values: {0->0.0, 1->+w_p, 2->-w_n}
        w_tile = tl.where(codes == 1, w_p, 0.0)
        w_tile = tl.where(codes == 2, neg_wn, w_tile)
        # w_tile: (16, BLOCK_N) float32 — decoded ternary weights

        # Load input tile: (BLOCK_M, 16) float32
        k_range = k0 + tl.arange(0, 16)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_range[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_range[None, :] < K),
            other=0.0
        )

        # Matrix multiply: (BLOCK_M, 16) @ (16, BLOCK_N) -> accumulate into (BLOCK_M, BLOCK_N)
        acc += tl.dot(x_tile, w_tile)

    # Store result
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc, mask=out_mask
    )


# ============================================================
# Triton Kernel: Ternary Matmul (Conditional Add/Subtract)
# ============================================================
# Classic ternary optimization: replaces N multiply-accumulate ops with
# N conditional additions + 2 final multiplications.

@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['M', 'N', 'K_packed'])
@triton.jit
def _ternary_matmul_cond_add_kernel(
    x_ptr, w_ptr, y_ptr, wp_ptr, wn_ptr,
    M, N, K, K_packed,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Ternary matmul via conditional add/subtract.

    Instead of: output[m,n] = sum_k(W[n,k] * X[m,k])
    Computes:   output[m,n] = w_p * sum(X[m,k] where W>0) - w_n * sum(X[m,k] where W<0)

    This replaces K multiply-accumulate operations with:
    - K conditional additions (add X to pos or neg accumulator based on code)
    - 2 final multiplications (scale by w_p and w_n)

    Zero weights are implicitly skipped (neither mask is active).
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc_pos = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_neg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    shifts = (tl.arange(0, 16) * 2).to(tl.int32)

    for ki in range(K_packed):
        k0 = ki * 16

        packed = tl.load(
            w_ptr + offs_n * stride_wn + ki,
            mask=offs_n < N, other=0
        )

        codes = (packed[None, :] >> shifts[:, None]) & 0x3  # (16, BLOCK_N)

        # Binary masks: 1.0 where code matches, 0.0 elsewhere
        pos_mask = (codes == 1).to(tl.float32)  # (16, BLOCK_N)
        neg_mask = (codes == 2).to(tl.float32)  # (16, BLOCK_N)

        k_range = k0 + tl.arange(0, 16)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_range[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_range[None, :] < K),
            other=0.0
        )

        # Conditional accumulation via dot product with binary masks
        # pos_sum += sum(X[m,k] * 1.0 where code=positive) = sum(X where positive)
        # neg_sum += sum(X[m,k] * 1.0 where code=negative) = sum(X where negative)
        acc_pos += tl.dot(x_tile, pos_mask)
        acc_neg += tl.dot(x_tile, neg_mask)

    # Final: only 2 multiplications per output element
    w_p = tl.load(wp_ptr)
    w_n = tl.load(wn_ptr)
    result = w_p * acc_pos - w_n * acc_neg

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        result, mask=out_mask
    )


# ============================================================
# Triton Kernel: Ternary Matmul (Int8 Signs — wider BLOCK_K)
# ============================================================
# Stores ternary weights as int8 {-1, 0, +1} (4x compression).
# Uses standard BLOCK_K (32/64/128) for tl.dot — better tensor core utilization.
# Combined with 2-bit packed storage for deployment (16x compression).

def encode_ternary_int8(W: torch.Tensor, w_p: float, w_n: float) -> torch.Tensor:
    """Encode ternary weights {-w_n, 0, +w_p} as int8 signs {-1, 0, +1}."""
    signs = torch.zeros(W.shape, dtype=torch.int8, device=W.device)
    signs[W > w_p * 0.5] = 1
    signs[W < -w_n * 0.5] = -1
    return signs.contiguous()


def decode_int8_to_fp32(signs: torch.Tensor, w_p: float, w_n: float) -> torch.Tensor:
    """Decode int8 signs back to FP32 ternary weights. For verification."""
    W = torch.zeros(signs.shape, dtype=torch.float32, device=signs.device)
    W[signs == 1] = w_p
    W[signs == -1] = -w_n
    return W


_INT8_CONFIGS = [
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
]


@triton.autotune(configs=_INT8_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _ternary_matmul_int8_kernel(
    x_ptr, w_ptr, y_ptr, wp_ptr, wn_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Ternary matmul with int8 sign storage.

    Weights stored as int8 {-1, 0, +1} (4x compression vs FP32).
    Uses standard BLOCK_K for tl.dot — better tensor core utilization than
    the 2-bit kernel (which is limited to K=16 per dot).

    Decode: signs -> {-w_n, 0, +w_p} via conditional moves in registers.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w_p = tl.load(wp_ptr)
    w_n = tl.load(wn_ptr)
    neg_wn = -w_n

    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Load int8 signs: (BLOCK_N, BLOCK_K) from (N, K) layout
        w_signs = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k_range[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_range[None, :] < K),
            other=0
        )  # (BLOCK_N, BLOCK_K) int8

        # Decode to weight values: {-1 -> -w_n, 0 -> 0.0, +1 -> +w_p}
        w_f32 = w_signs.to(tl.float32)
        w_decoded = tl.where(w_f32 > 0.5, w_p, tl.where(w_f32 < -0.5, neg_wn, 0.0))
        # Transpose: (BLOCK_N, BLOCK_K) -> (BLOCK_K, BLOCK_N)
        w_t = tl.trans(w_decoded)

        # Load input: (BLOCK_M, BLOCK_K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_range[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_range[None, :] < K),
            other=0.0
        )

        # Matmul: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(x_tile, w_t)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc, mask=out_mask
    )


# ============================================================
# Python Wrappers
# ============================================================

def ternary_matmul(X: torch.Tensor, W_packed: torch.Tensor,
                   w_p: torch.Tensor, w_n: torch.Tensor, K: int,
                   method: str = 'decode',
                   W_int8: torch.Tensor = None) -> torch.Tensor:
    """
    Ternary matrix multiplication: Y = X @ decode(W)^T

    Args:
        X: (M, K) float32 input activations, contiguous
        W_packed: (N, K_packed) int32 packed ternary codes, contiguous
            (used for 'decode' and 'cond_add' methods)
        w_p: scalar float32 tensor — positive scaling factor
        w_n: scalar float32 tensor — negative scaling factor
        K: original (unpacked) K dimension
        method: 'decode' | 'cond_add' | 'int8'
        W_int8: (N, K) int8 signs {-1, 0, +1} — required for method='int8'

    Returns:
        Y: (M, N) float32
    """
    assert X.is_cuda
    assert X.dtype == torch.float32
    X = X.contiguous()

    if method == 'int8':
        assert W_int8 is not None, "W_int8 required for method='int8'"
        W_int8 = W_int8.contiguous()
        M = X.shape[0]
        N = W_int8.shape[0]
        Y = torch.empty(M, N, dtype=torch.float32, device=X.device)
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )
        _ternary_matmul_int8_kernel[grid](
            X, W_int8, Y, w_p, w_n,
            M, N, K,
            X.stride(0), X.stride(1),
            W_int8.stride(0), W_int8.stride(1),
            Y.stride(0), Y.stride(1),
        )
        return Y

    assert W_packed is not None and W_packed.dtype == torch.int32
    W_packed = W_packed.contiguous()
    M = X.shape[0]
    N, K_packed = W_packed.shape
    Y = torch.empty(M, N, dtype=torch.float32, device=X.device)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    kernel = _ternary_matmul_decode_kernel if method == 'decode' else _ternary_matmul_cond_add_kernel
    kernel[grid](
        X, W_packed, Y, w_p, w_n,
        M, N, K, K_packed,
        X.stride(0), X.stride(1),
        W_packed.stride(0), W_packed.stride(1),
        Y.stride(0), Y.stride(1),
    )
    return Y


def ternary_conv2d(input: torch.Tensor, W_packed: torch.Tensor,
                   w_p: torch.Tensor, w_n: torch.Tensor,
                   K: int, C_out: int, kernel_size: tuple,
                   stride: int = 1, padding: int = 0,
                   bias: torch.Tensor = None,
                   method: str = 'decode') -> torch.Tensor:
    """
    Ternary convolution: im2col + ternary matmul.

    Args:
        input: (B, C_in, H, W) float32
        W_packed: (C_out, K_packed) int32 packed ternary codes
            where K = C_in * kH * kW
        w_p, w_n: scalar float32 tensors
        K: C_in * kH * kW (original unpacked K)
        C_out: number of output channels
        kernel_size: (kH, kW) tuple
        stride: convolution stride
        padding: convolution padding
        bias: optional (C_out,) float32 bias
        method: 'decode' or 'cond_add'

    Returns:
        output: (B, C_out, H_out, W_out) float32
    """
    B, C_in, H, W_in = input.shape
    kH, kW = kernel_size

    # im2col: (B, C_in*kH*kW, L) where L = H_out * W_out
    unfolded = F.unfold(input, kernel_size, stride=stride, padding=padding)
    L = unfolded.shape[2]

    # Reshape to (B*L, K) for matmul
    X = unfolded.permute(0, 2, 1).reshape(B * L, K).contiguous()

    # Ternary matmul: (B*L, K) @ decode(W_packed)^T = (B*L, C_out)
    Y = ternary_matmul(X, W_packed, w_p, w_n, K, method=method)

    # Add bias
    if bias is not None:
        Y = Y + bias.unsqueeze(0)

    # Reshape to (B, C_out, H_out, W_out)
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1
    output = Y.reshape(B, L, C_out).permute(0, 2, 1).reshape(B, C_out, H_out, W_out)

    return output
