"""
Benchmark: Sparse Ternary Kernel vs All Baselines
===================================================

Compares the sparse ternary matmul kernel against:
  1. torch.mm        — dense FP32 matmul (cuBLAS, TF32 disabled)
  2. torch.sparse.mm — generic sparse matmul (cuSPARSE, CSR format)
  3. 2-bit dense      — existing packed ternary kernel (from ternary_matmul.py)
  4. Sparse ternary   — THIS WORK: gather-accumulate with multiply-free inner loop

Sweeps sparsity levels (0%–95%) across ResNet-50 layer shapes at multiple batch sizes.
Uses CUDA event timing with warmup for rigorous measurement.

Usage:
    source env/bin/activate
    cd /workspace/EMA-pTTQ
    python src/triton_kernels/benchmark_sparse.py [--device cuda:1]
"""

import torch
import argparse
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))
from ternary_matmul import pack_ternary_weights, ternary_matmul, encode_ternary_int8
from sparse_ternary_matmul import (
    prepare_sparse_ternary, sparse_ternary_matmul, sparse_ternary_matmul_preT,
    sparse_format_memory_bytes,
)



# ============================================================
# Utilities
# ============================================================

def create_ternary_weights(N, K, w_p, w_n, sparsity, device):
    """Create random ternary weight matrix with exact sparsity."""
    W = torch.zeros(N, K, dtype=torch.float32, device=device)
    rand = torch.rand(N, K, device=device)
    nz = 1.0 - sparsity
    W[rand < nz / 2] = w_p
    W[rand > 1 - nz / 2] = -w_n
    return W


def bench_fn(fn, num_warmup=50, num_trials=200):
    """Benchmark function using CUDA event timing, returns time in microseconds."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_trials):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / num_trials  # ms -> us


# ============================================================
# Single benchmark
# ============================================================

def benchmark_one(M, N, K, w_p, w_n, sparsity, device,
                  num_warmup=50, num_trials=200, skip_sparse_mm=False):
    """
    Benchmark all methods for one (shape, sparsity) combination.

    Returns dict with timing (us), speedups, errors.
    """
    W = create_ternary_weights(N, K, w_p, w_n, sparsity, device)
    X = torch.randn(M, K, dtype=torch.float32, device=device)

    # Actual sparsity
    actual_sparsity = (W == 0).float().mean().item()

    # Reference
    Y_ref = torch.mm(X, W.T)

    # ---- 1. torch.mm (dense FP32) ----
    t_mm = bench_fn(lambda: torch.mm(X, W.T), num_warmup, num_trials)

    # ---- 2. torch.sparse.mm (cuSPARSE CSR) ----
    t_sparse_mm = float('nan')
    err_sparse_mm = float('nan')
    if not skip_sparse_mm:
        try:
            W_csr = W.to_sparse_csr()
            X_T_dense = X.T.contiguous()
            # torch.sparse.mm: (N,K)_sparse @ (K,M)_dense = (N,M)
            Y_sp = torch.sparse.mm(W_csr, X_T_dense).T
            err_sparse_mm = (Y_ref - Y_sp).abs().max().item()
            t_sparse_mm = bench_fn(lambda: torch.sparse.mm(W_csr, X_T_dense),
                                   num_warmup, num_trials)
        except Exception as e:
            print(f"    [WARN] torch.sparse.mm failed: {e}")

    # ---- 3. 2-bit dense kernel ----
    W_packed = pack_ternary_weights(W, w_p, w_n)
    w_p_t = torch.tensor(w_p, dtype=torch.float32, device=device)
    w_n_t = torch.tensor(w_n, dtype=torch.float32, device=device)
    Y_2bit = ternary_matmul(X, W_packed, w_p_t, w_n_t, K, method='decode')
    err_2bit = (Y_ref - Y_2bit).abs().max().item()
    t_2bit = bench_fn(lambda: ternary_matmul(X, W_packed, w_p_t, w_n_t, K, method='decode'),
                      num_warmup, num_trials)

    # ---- 4. Sparse ternary kernel (with X transpose in timing) ----
    sparse_w = prepare_sparse_ternary(W, w_p, w_n)
    Y_sparse_tern = sparse_ternary_matmul(X, sparse_w, w_p, w_n)
    err_sparse_tern = (Y_ref - Y_sparse_tern).abs().max().item()
    t_sparse_tern = bench_fn(
        lambda: sparse_ternary_matmul(X, sparse_w, w_p, w_n),
        num_warmup, num_trials)

    # ---- 4b. Sparse ternary kernel (pre-transposed, kernel-only timing) ----
    X_T_pre = X.T.contiguous()
    wp1 = torch.tensor([w_p], dtype=torch.float32, device=device)
    wn1 = torch.tensor([w_n], dtype=torch.float32, device=device)
    t_sparse_tern_noT = bench_fn(
        lambda: sparse_ternary_matmul_preT(X_T_pre, sparse_w, wp1, wn1, M),
        num_warmup, num_trials)

    return {
        'M': M, 'N': N, 'K': K,
        'sparsity': actual_sparsity,
        't_mm': t_mm,
        't_sparse_mm': t_sparse_mm,
        't_2bit': t_2bit,
        't_ours': t_sparse_tern,
        't_ours_noT': t_sparse_tern_noT,
        'speedup_vs_mm': t_mm / t_sparse_tern,
        'speedup_vs_mm_noT': t_mm / t_sparse_tern_noT,
        'speedup_vs_sparse_mm': t_sparse_mm / t_sparse_tern_noT if not math.isnan(t_sparse_mm) else float('nan'),
        'speedup_vs_2bit': t_2bit / t_sparse_tern_noT,
        'err_sparse_mm': err_sparse_mm,
        'err_2bit': err_2bit,
        'err_ours': err_sparse_tern,
        'nnz_pos': sparse_w['pos_count'].float().mean().item(),
        'nnz_neg': sparse_w['neg_count'].float().mean().item(),
    }


# ============================================================
# Main benchmark suite
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark sparse ternary kernel')
    parser.add_argument('--device', default='cuda:1', help='CUDA device')
    parser.add_argument('--quick', action='store_true', help='Quick mode: fewer shapes/sparsities')
    args = parser.parse_args()

    device = args.device
    torch.cuda.set_device(device)

    # CRITICAL: Disable TF32 for fair FP32 comparison
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"PyTorch: {torch.__version__}")
    print(f"TF32: DISABLED (fair FP32 comparison)")
    print()

    w_p, w_n = 0.5, 0.3

    # ResNet-50 matmul shapes: (name, M, N, K)
    if args.quick:
        shapes = [
            ("B1 layer4 1x1",      4, 2048,  512),
            ("B32 layer3 1x1",   512, 1024,  256),
            ("B32 layer4 1x1",   128, 2048,  512),
        ]
        sparsities = [0.0, 0.6, 0.8, 0.95]
    else:
        shapes = [
            # Batch=1 inference
            ("B1 layer1 1x1",    256,  256,   64),
            ("B1 layer3 1x1",     16, 1024,  256),
            ("B1 layer4 1x1",      4, 2048,  512),
            ("B1 layer4 3x3",      4,  512, 4608),
            # Batch=32
            ("B32 layer3 1x1",   512, 1024,  256),
            ("B32 layer4 1x1",   128, 2048,  512),
            ("B32 early 3x3",  8192,  128,  576),
            # Batch=128
            ("B128 layer4 1x1",  512, 2048,  512),
        ]
        sparsities = [0.0, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95]

    # ================================================================
    # Main benchmark loop
    # ================================================================
    all_results = []

    for name, M, N, K in shapes:
        print("=" * 140)
        print(f"Shape: {name}  (M={M}, N={N}, K={K})")
        print("=" * 140)
        print(f"{'Sparsity':>8} | "
              f"{'torch.mm':>10} {'sparse.mm':>10} {'2-bit':>10} {'Ours':>10} {'Ours(noT)':>10} | "
              f"{'vs mm':>7} {'vs mm*':>7} {'vs sp':>7} {'vs 2b':>7} | "
              f"{'Err':>8} {'nnz_p':>6} {'nnz_n':>6}")
        print(f"{'':>8} | "
              f"{'(us)':>10} {'(us)':>10} {'(us)':>10} {'(us)':>10} {'(us)':>10} | "
              f"{'':>7} {'(noT)':>7} {'(noT)':>7} {'(noT)':>7} | "
              f"{'':>8} {'':>6} {'':>6}")
        print("-" * 140)

        for sp in sparsities:
            r = benchmark_one(M, N, K, w_p, w_n, sp, device)
            r['name'] = name
            all_results.append(r)

            # Format speedups
            vs_mm = f"{r['speedup_vs_mm']:.2f}x"
            vs_mm_noT = f"{r['speedup_vs_mm_noT']:.2f}x"
            vs_sp = f"{r['speedup_vs_sparse_mm']:.2f}x" if not math.isnan(r['speedup_vs_sparse_mm']) else "N/A"
            vs_2b = f"{r['speedup_vs_2bit']:.2f}x"

            marker = ''
            if r['speedup_vs_mm_noT'] > 2.0:
                marker = ' ***'
            elif r['speedup_vs_mm_noT'] > 1.0:
                marker = ' *'

            print(f"{r['sparsity']:>7.1%} | "
                  f"{r['t_mm']:>10.1f} {r['t_sparse_mm']:>10.1f} {r['t_2bit']:>10.1f} "
                  f"{r['t_ours']:>10.1f} {r['t_ours_noT']:>10.1f} | "
                  f"{vs_mm:>7} {vs_mm_noT:>7} {vs_sp:>7} {vs_2b:>7} | "
                  f"{r['err_ours']:.1e} {r['nnz_pos']:>6.0f} {r['nnz_neg']:>6.0f}"
                  f"{marker}")

        print()

    # ================================================================
    # Summary tables
    # ================================================================
    print()
    print("=" * 100)
    print("SUMMARY: Speedup vs torch.mm (kernel-only, no transpose overhead)")
    print("=" * 100)

    # Header: sparsity levels
    sp_strs = [f"{sp:.0%}" for sp in sparsities]
    header = f"{'Shape':<22} | " + " | ".join(f"{s:>7}" for s in sp_strs)
    print(header)
    print("-" * len(header))

    for name, M, N, K in shapes:
        row = f"{name:<22} |"
        for sp in sparsities:
            matches = [r for r in all_results
                       if r['name'] == name and abs(r['sparsity'] - sp) < 0.05]
            if matches:
                row += f" {matches[0]['speedup_vs_mm_noT']:>6.2f}x |"
            else:
                row += f"    N/A |"
        print(row)

    # Speedup vs torch.sparse.mm
    print()
    print("=" * 100)
    print("SUMMARY: Speedup vs torch.sparse.mm (kernel-only)")
    print("=" * 100)
    print(header)
    print("-" * len(header))

    for name, M, N, K in shapes:
        row = f"{name:<22} |"
        for sp in sparsities:
            matches = [r for r in all_results
                       if r['name'] == name and abs(r['sparsity'] - sp) < 0.05]
            if matches and not math.isnan(matches[0]['speedup_vs_sparse_mm']):
                row += f" {matches[0]['speedup_vs_sparse_mm']:>6.2f}x |"
            else:
                row += f"    N/A |"
        print(row)

    # Speedup vs 2-bit dense
    print()
    print("=" * 100)
    print("SUMMARY: Speedup vs 2-bit dense kernel (kernel-only)")
    print("=" * 100)
    print(header)
    print("-" * len(header))

    for name, M, N, K in shapes:
        row = f"{name:<22} |"
        for sp in sparsities:
            matches = [r for r in all_results
                       if r['name'] == name and abs(r['sparsity'] - sp) < 0.05]
            if matches:
                row += f" {matches[0]['speedup_vs_2bit']:>6.2f}x |"
            else:
                row += f"    N/A |"
        print(row)

    # Correctness summary
    print()
    print("=" * 60)
    print("CORRECTNESS (max absolute error vs torch.mm reference)")
    print("=" * 60)
    max_err = max(r['err_ours'] for r in all_results)
    max_err_2bit = max(r['err_2bit'] for r in all_results)
    max_err_sp = max((r['err_sparse_mm'] for r in all_results if not math.isnan(r['err_sparse_mm'])), default=float('nan'))
    print(f"  Sparse ternary kernel: {max_err:.2e}")
    print(f"  2-bit dense kernel:    {max_err_2bit:.2e}")
    print(f"  torch.sparse.mm:       {max_err_sp:.2e}")

    # Key findings
    print()
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Find shapes where we beat torch.mm at 60% sparsity
    wins_60 = [(r['name'], r['speedup_vs_mm_noT'])
               for r in all_results if abs(r['sparsity'] - 0.6) < 0.05]
    beats_mm_60 = [(n, s) for n, s in wins_60 if s > 1.0]
    print(f"\nShapes where sparse ternary beats torch.mm at ~60% sparsity:")
    for name, spd in sorted(beats_mm_60, key=lambda x: -x[1]):
        print(f"  {name:<22} {spd:.2f}x")
    if not beats_mm_60:
        print("  (none)")

    # Scaling with sparsity check
    print(f"\nSparsity scaling (does speedup increase with sparsity?):")
    for name, M, N, K in shapes:
        shape_results = [(r['sparsity'], r['speedup_vs_mm_noT'])
                         for r in all_results if r['name'] == name]
        shape_results.sort()
        if len(shape_results) >= 3:
            low_sp = shape_results[0][1]
            high_sp = shape_results[-1][1]
            scales = "YES" if high_sp > low_sp * 1.2 else "NO"
            trend = " -> ".join(f"{sp:.0%}:{spd:.2f}x" for sp, spd in shape_results)
            print(f"  {name:<22} {scales:4}  {trend}")


if __name__ == '__main__':
    main()
