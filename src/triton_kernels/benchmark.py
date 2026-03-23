"""
Benchmark: Ternary Triton Kernel vs PyTorch torch.mm / F.conv2d

Measures wall-clock time and GPU memory for ternary-quantized matmul/conv2d
using realistic tensor shapes from ResNet-50 on CIFAR-10 / TinyImageNet.

Usage:
    python src/triton_kernels/benchmark.py [--device cuda:1]
"""

import torch
import torch.nn.functional as F
import argparse
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ternary_matmul import (
    pack_ternary_weights, unpack_ternary_weights, ternary_matmul, ternary_conv2d,
    encode_ternary_int8, decode_int8_to_fp32,
    weight_memory_bytes, fp32_memory_bytes,
)


def create_ternary_weights(N, K, w_p, w_n, sparsity=0.6, device='cuda'):
    """Create a random ternary weight matrix with given sparsity."""
    W = torch.zeros(N, K, dtype=torch.float32, device=device)
    rand = torch.rand(N, K, device=device)
    nonzero_frac = 1.0 - sparsity
    W[rand < nonzero_frac / 2] = w_p
    W[rand > 1 - nonzero_frac / 2] = -w_n
    return W


def _bench(fn, num_warmup=50, num_trials=200):
    """Benchmark a function, return time in microseconds."""
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
    return start.elapsed_time(end) * 1000 / num_trials  # microseconds


def benchmark_matmul(M, N, K, w_p=0.5, w_n=0.3, sparsity=0.6, device='cuda',
                     num_warmup=50, num_trials=200):
    """Benchmark all ternary matmul variants vs torch.mm."""
    W = create_ternary_weights(N, K, w_p, w_n, sparsity, device)
    X = torch.randn(M, K, dtype=torch.float32, device=device)

    W_packed = pack_ternary_weights(W, w_p, w_n)
    W_int8 = encode_ternary_int8(W, w_p, w_n)
    w_p_t = torch.tensor(w_p, dtype=torch.float32, device=device)
    w_n_t = torch.tensor(w_n, dtype=torch.float32, device=device)

    # Verify correctness
    Y_ref = torch.mm(X, W.T)

    # Pack roundtrip
    W_unp = unpack_ternary_weights(W_packed, w_p, w_n, K)
    pack_err = (W - W_unp).abs().max().item()

    Y_dec = ternary_matmul(X, W_packed, w_p_t, w_n_t, K, method='decode')
    err_decode = (Y_ref - Y_dec).abs().max().item()

    Y_cond = ternary_matmul(X, W_packed, w_p_t, w_n_t, K, method='cond_add')
    err_cond = (Y_ref - Y_cond).abs().max().item()

    Y_int8 = ternary_matmul(X, None, w_p_t, w_n_t, K, method='int8', W_int8=W_int8)
    err_int8 = (Y_ref - Y_int8).abs().max().item()

    # Benchmark
    t_torch = _bench(lambda: torch.mm(X, W.T), num_warmup, num_trials)
    t_decode = _bench(lambda: ternary_matmul(X, W_packed, w_p_t, w_n_t, K, method='decode'),
                      num_warmup, num_trials)
    t_cond = _bench(lambda: ternary_matmul(X, W_packed, w_p_t, w_n_t, K, method='cond_add'),
                    num_warmup, num_trials)
    t_int8 = _bench(lambda: ternary_matmul(X, None, w_p_t, w_n_t, K, method='int8', W_int8=W_int8),
                    num_warmup, num_trials)

    mem_fp32 = fp32_memory_bytes(N, K)
    mem_packed = weight_memory_bytes(W_packed)
    mem_int8 = W_int8.numel() * W_int8.element_size()

    return {
        'M': M, 'N': N, 'K': K,
        't_torch_us': t_torch,
        't_decode_us': t_decode,
        't_cond_us': t_cond,
        't_int8_us': t_int8,
        'speedup_decode': t_torch / t_decode,
        'speedup_cond': t_torch / t_cond,
        'speedup_int8': t_torch / t_int8,
        'mem_fp32': mem_fp32,
        'mem_packed': mem_packed,
        'mem_int8': mem_int8,
        'pack_err': pack_err,
        'err_decode': err_decode,
        'err_cond': err_cond,
        'err_int8': err_int8,
    }


def benchmark_conv2d(B, C_in, C_out, H, W_in, kH, kW, stride, padding,
                     w_p=0.5, w_n=0.3, sparsity=0.6, device='cuda',
                     num_warmup=20, num_trials=100):
    """Benchmark ternary conv2d vs F.conv2d."""
    K = C_in * kH * kW
    W_conv = create_ternary_weights(C_out, K, w_p, w_n, sparsity, device)
    W_conv_4d = W_conv.reshape(C_out, C_in, kH, kW)
    input_t = torch.randn(B, C_in, H, W_in, dtype=torch.float32, device=device)

    W_packed = pack_ternary_weights(W_conv, w_p, w_n)
    w_p_t = torch.tensor(w_p, dtype=torch.float32, device=device)
    w_n_t = torch.tensor(w_n, dtype=torch.float32, device=device)

    Y_ref = F.conv2d(input_t, W_conv_4d, stride=stride, padding=padding)
    Y_ternary = ternary_conv2d(
        input_t, W_packed, w_p_t, w_n_t, K, C_out, (kH, kW),
        stride=stride, padding=padding, method='decode'
    )
    conv_err = (Y_ref - Y_ternary).abs().max().item()

    t_conv = _bench(lambda: F.conv2d(input_t, W_conv_4d, stride=stride, padding=padding),
                    num_warmup, num_trials)
    t_ternary = _bench(
        lambda: ternary_conv2d(input_t, W_packed, w_p_t, w_n_t, K, C_out, (kH, kW),
                               stride=stride, padding=padding, method='decode'),
        num_warmup, num_trials)

    mem_fp32 = fp32_memory_bytes(C_out, K)
    mem_packed = weight_memory_bytes(W_packed)

    return {
        'shape': f'Conv({C_out},{C_in},{kH}x{kW}) B={B} H={H}',
        't_conv_us': t_conv,
        't_ternary_us': t_ternary,
        'speedup': t_conv / t_ternary,
        'conv_err': conv_err,
        'mem_fp32': mem_fp32,
        'mem_packed': mem_packed,
        'mem_ratio': mem_fp32 / mem_packed,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark ternary Triton kernels')
    parser.add_argument('--device', default='cuda:1', help='CUDA device')
    args = parser.parse_args()

    device = args.device
    torch.cuda.set_device(device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ============================================================
    # Matmul Benchmarks
    # ============================================================
    matmul_shapes = [
        # Batch=1 inference (TinyImageNet ResNet-50)
        ("B1 layer1 1x1",    256,  256,   64),
        ("B1 layer1 3x3",    256,   64,  576),
        ("B1 layer2 1x1",     64,  512,  128),
        ("B1 layer2 3x3",     64,  128, 1152),
        ("B1 layer3 1x1",     16, 1024,  256),
        ("B1 layer3 3x3",     16,  256, 2304),
        ("B1 layer4 1x1",      4, 2048,  512),
        ("B1 layer4 3x3",      4,  512, 4608),
        # Batch=32
        ("B32 layer3 1x1",  512, 1024,  256),
        ("B32 layer4 1x1",  128, 2048,  512),
        ("B32 layer4 3x3",  128,  512, 4608),
        # Batch=128
        ("B128 layer4 1x1", 512, 2048,  512),
        # Large early layer
        ("B32 early 3x3",  8192,  128,  576),
    ]

    print("=" * 130)
    print("MATMUL BENCHMARKS: Ternary Triton Kernels vs torch.mm")
    print("=" * 130)
    print(f"{'Name':<22} {'M':>5} {'N':>5} {'K':>5} | "
          f"{'torch.mm':>9} {'Decode':>9} {'CondAdd':>9} {'Int8':>9} | "
          f"{'Dec':>6} {'CA':>6} {'I8':>6} | "
          f"{'FP32':>7} {'2bit':>7} {'I8':>7}")
    print(f"{'':22} {'':>5} {'':>5} {'':>5} | "
          f"{'(us)':>9} {'(us)':>9} {'(us)':>9} {'(us)':>9} | "
          f"{'':>6} {'':>6} {'':>6} | "
          f"{'(KB)':>7} {'(KB)':>7} {'(KB)':>7}")
    print("-" * 130)

    results = []
    for name, M, N, K in matmul_shapes:
        r = benchmark_matmul(M, N, K, device=device)
        results.append(r)
        best_spd = max(r['speedup_decode'], r['speedup_cond'], r['speedup_int8'])
        marker = ' ***' if best_spd > 1.5 else (' **' if best_spd > 1.1 else '')
        print(f"{name:<22} {M:>5} {N:>5} {K:>5} | "
              f"{r['t_torch_us']:>9.1f} {r['t_decode_us']:>9.1f} {r['t_cond_us']:>9.1f} {r['t_int8_us']:>9.1f} | "
              f"{r['speedup_decode']:>5.2f}x {r['speedup_cond']:>5.2f}x {r['speedup_int8']:>5.2f}x | "
              f"{r['mem_fp32']/1024:>7.1f} {r['mem_packed']/1024:>7.1f} {r['mem_int8']/1024:>7.1f}"
              f"{marker}")

    print()
    print("Correctness (max abs error across all shapes):")
    print(f"  Pack roundtrip:  {max(r['pack_err'] for r in results):.2e}")
    print(f"  Decode kernel:   {max(r['err_decode'] for r in results):.2e}")
    print(f"  Cond-add kernel: {max(r['err_cond'] for r in results):.2e}")
    print(f"  Int8 kernel:     {max(r['err_int8'] for r in results):.2e}")

    # Summary: best kernel per shape
    print()
    print("Best ternary kernel per shape:")
    total_torch = 0
    total_best = 0
    for r in results:
        best_method = 'decode'
        best_time = r['t_decode_us']
        if r['t_cond_us'] < best_time:
            best_method = 'cond_add'
            best_time = r['t_cond_us']
        if r['t_int8_us'] < best_time:
            best_method = 'int8'
            best_time = r['t_int8_us']
        speedup = r['t_torch_us'] / best_time
        total_torch += r['t_torch_us']
        total_best += best_time
        name = f"M={r['M']},N={r['N']},K={r['K']}"
        print(f"  {name:<30} best={best_method:<8} {speedup:.2f}x  ({r['t_torch_us']:.1f} -> {best_time:.1f} us)")
    print(f"  {'TOTAL':<30} {'':8} {total_torch/total_best:.2f}x  ({total_torch:.0f} -> {total_best:.0f} us)")

    # ============================================================
    # Conv2d Benchmarks
    # ============================================================
    conv_shapes = [
        (1,  64,  64,  16, 16, 3, 3, 1, 1),
        (1, 128, 128,   8,  8, 3, 3, 1, 1),
        (1, 256, 256,   4,  4, 3, 3, 1, 1),
        (1, 512, 512,   2,  2, 3, 3, 1, 1),
        (1, 256, 1024,  4,  4, 1, 1, 1, 0),
        (1, 512, 2048,  2,  2, 1, 1, 1, 0),
        (32, 256, 1024, 4, 4, 1, 1, 1, 0),
        (32, 512, 2048, 2, 2, 1, 1, 1, 0),
    ]

    print()
    print("=" * 100)
    print("CONV2D BENCHMARKS: Ternary im2col+matmul vs F.conv2d (cuDNN)")
    print("=" * 100)
    print(f"{'Shape':<40} | {'F.conv2d':>10} {'Ternary':>10} | "
          f"{'Speedup':>8} | {'FP32':>8} {'Packed':>8} {'Ratio':>6} | {'Err':>10}")
    print("-" * 100)

    for B, C_in, C_out, H, W_in, kH, kW, stride, padding in conv_shapes:
        r = benchmark_conv2d(B, C_in, C_out, H, W_in, kH, kW, stride, padding, device=device)
        print(f"{r['shape']:<40} | {r['t_conv_us']:>10.1f} {r['t_ternary_us']:>10.1f} | "
              f"{r['speedup']:>7.2f}x | "
              f"{r['mem_fp32']/1024:>8.1f} {r['mem_packed']/1024:>8.1f} {r['mem_ratio']:>5.1f}x | "
              f"{r['conv_err']:.2e}")

    # ============================================================
    # Memory: Full ResNet-50
    # ============================================================
    print()
    print("=" * 60)
    print("MEMORY: Full ResNet-50 weight storage comparison")
    print("=" * 60)

    resnet50_layers = [
        (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1),
        (64, 256, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1),
        (64, 256, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1),
        (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1),
        (128, 512, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1),
        (128, 512, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1),
        (128, 512, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1),
        (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
        (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
        (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
        (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
        (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
        (256, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1),
        (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1),
        (512, 2048, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1),
        (512, 2048, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1),
        (256, 64, 1, 1), (512, 256, 1, 1), (1024, 512, 1, 1), (2048, 1024, 1, 1),
    ]

    total_fp32 = 0
    total_packed = 0
    total_int8 = 0
    for C_out, C_in, kH, kW in resnet50_layers:
        K = C_in * kH * kW
        total_fp32 += C_out * K * 4
        total_packed += C_out * math.ceil(K / 16) * 4 + 8
        total_int8 += C_out * K + 8

    print(f"  FP32 weights:     {total_fp32 / 1024 / 1024:.2f} MB")
    print(f"  2-bit packed:     {total_packed / 1024 / 1024:.2f} MB  ({total_fp32/total_packed:.1f}x compression)")
    print(f"  Int8 signs:       {total_int8 / 1024 / 1024:.2f} MB  ({total_fp32/total_int8:.1f}x compression)")
    print(f"  Memory saved (2-bit): {(total_fp32 - total_packed) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
