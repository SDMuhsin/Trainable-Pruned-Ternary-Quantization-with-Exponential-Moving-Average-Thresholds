#!/usr/bin/env python3
"""
Diagnostic: Compare weight distributions of ConvNeXt vs ResNet-50 FP models
and compute what EMA-pTTQ thresholds / sparsity would be at initialization.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.utils.model_compression import get_params_groups_to_quantize, pruning_function_pTTQ_experimental

def load_model_and_params(model_path, model_to_use):
    """Load FP model and get quantizable parameters."""
    if model_to_use == 'tinyimagenetconvnext':
        from src.Models.CNNs.convnext import ConvNeXtTinyClassificationModel
        model = ConvNeXtTinyClassificationModel(num_classes=200)
    elif model_to_use == 'tinyimagenetresnet50':
        from src.Models.CNNs.resnet50 import ResNet50ClassificationModel
        model = ResNet50ClassificationModel(nb_classes=200)
    else:
        raise ValueError(f"Unknown model: {model_to_use}")

    data = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    params, names = get_params_groups_to_quantize(model, model_to_use)
    return model, params, names


def analyze_model(model_path, model_to_use, alpha=10000.0, init_x=1.0, init_y=1.0, k=1.0):
    """Analyze weight distributions and compute initial EMA-pTTQ thresholds."""
    model, params, names = load_model_and_params(model_path, model_to_use)

    weights_to_quantize = params['ToQuantize']['params']

    print(f"\n{'='*100}")
    print(f"MODEL: {model_to_use} ({model_path})")
    print(f"Quantizable layers: {len(names)}")
    print(f"Alpha={alpha}, init_x={init_x}, init_y={init_y}, k={k}")
    print(f"{'='*100}")

    total_params = 0
    total_zero = 0

    layer_data = []

    for i, (name, w) in enumerate(zip(names, weights_to_quantize)):
        w_data = w.data
        numel = w_data.numel()
        total_params += numel

        w_mean = w_data.mean().item()
        w_std = w_data.std().item()
        w_abs_mean = w_data.abs().mean().item()
        w_min = w_data.min().item()
        w_max = w_data.max().item()

        # Compute EMA thresholds (first call = no EMA, just raw)
        delta_min = abs(w_mean + init_x * w_std)
        delta_max = abs(w_mean + init_y * w_std)

        # Compute sparsity with these thresholds
        # The pruning function zeros weights where |w| < delta (approximately)
        # With alpha=10000 and k=1: weight is zero if -delta_min < w < delta_max
        sparsity_count = ((w_data > -k * delta_min) & (w_data < k * delta_max)).sum().item()
        sparsity = sparsity_count / numel
        total_zero += sparsity_count

        # Detect layer type
        if 'dwconv' in name or ('features' in name and w_data.ndim == 4 and w_data.shape[1] == 1):
            # Depthwise conv: groups == in_channels, so dim 1 == 1
            if w_data.ndim == 4 and w_data.shape[1] == 1:
                layer_type = "DEPTHWISE"
            elif 'dwconv' in name:
                layer_type = "DEPTHWISE"
            else:
                layer_type = "CONV"
        elif w_data.ndim == 2:
            layer_type = "LINEAR"
        elif w_data.ndim == 4:
            if w_data.shape[2] == 1 and w_data.shape[3] == 1:
                layer_type = "CONV1x1"
            else:
                layer_type = f"CONV{w_data.shape[2]}x{w_data.shape[3]}"
        else:
            layer_type = f"OTHER({w_data.ndim}D)"

        layer_data.append({
            'name': name, 'type': layer_type, 'shape': list(w_data.shape),
            'numel': numel, 'mean': w_mean, 'std': w_std, 'abs_mean': w_abs_mean,
            'min': w_min, 'max': w_max, 'delta_min': delta_min, 'delta_max': delta_max,
            'sparsity': sparsity
        })

    # Print summary by layer type
    print(f"\n--- Per-layer statistics ---")
    print(f"{'Layer':<60s} {'Type':<10s} {'Shape':<20s} {'Numel':>8s} {'Mean':>8s} {'Std':>8s} {'|Mean|':>8s} {'Delta_min':>10s} {'Delta_max':>10s} {'Sparsity':>8s}")
    print("-" * 160)

    for d in layer_data:
        print(f"{d['name']:<60s} {d['type']:<10s} {str(d['shape']):<20s} {d['numel']:>8d} {d['mean']:>8.5f} {d['std']:>8.5f} {d['abs_mean']:>8.5f} {d['delta_min']:>10.5f} {d['delta_max']:>10.5f} {d['sparsity']:>7.1%}")

    overall_sparsity = total_zero / total_params
    print(f"\n--- Overall: {total_zero}/{total_params} = {overall_sparsity:.1%} sparsity ---")

    # Summary by layer type
    type_stats = {}
    for d in layer_data:
        t = d['type']
        if t not in type_stats:
            type_stats[t] = {'count': 0, 'params': 0, 'zero': 0, 'stds': [], 'sparsities': [], 'deltas': []}
        type_stats[t]['count'] += 1
        type_stats[t]['params'] += d['numel']
        type_stats[t]['zero'] += int(d['sparsity'] * d['numel'])
        type_stats[t]['stds'].append(d['std'])
        type_stats[t]['sparsities'].append(d['sparsity'])
        type_stats[t]['deltas'].append(d['delta_max'])

    print(f"\n--- Summary by layer type ---")
    print(f"{'Type':<12s} {'Count':>6s} {'Params':>10s} {'Avg Std':>10s} {'Avg Sparsity':>13s} {'Avg Delta':>10s}")
    for t, s in sorted(type_stats.items()):
        avg_sparsity = s['zero'] / s['params'] if s['params'] > 0 else 0
        print(f"{t:<12s} {s['count']:>6d} {s['params']:>10d} {np.mean(s['stds']):>10.5f} {avg_sparsity:>12.1%} {np.mean(s['deltas']):>10.5f}")

    # Also run the actual pruning function to see what it produces
    print(f"\n--- Actual pruning_function_pTTQ_experimental output ---")
    # Reset EMA states
    if hasattr(pruning_function_pTTQ_experimental, 'ema_states'):
        del pruning_function_pTTQ_experimental.ema_states

    total_actual_zero = 0
    for i, (name, w) in enumerate(zip(names, weights_to_quantize)):
        w_data = w.data
        pruned = pruning_function_pTTQ_experimental(w_data, alpha, init_x, init_y, k=k, beta=0.9, layer_id=i)
        eps = 1e-6
        n_zero = ((pruned.abs() <= eps)).sum().item()
        n_pos = (pruned > eps).sum().item()
        n_neg = (pruned < -eps).sum().item()
        actual_sparsity = n_zero / w_data.numel()
        total_actual_zero += n_zero

        layer_type = layer_data[i]['type']
        print(f"  [{layer_type:<10s}] {name:<60s} sparsity={actual_sparsity:.1%} (+{n_pos} 0={n_zero} -{n_neg})")

    actual_overall = total_actual_zero / total_params
    print(f"\n  Overall actual sparsity: {actual_overall:.1%}")

    return layer_data


if __name__ == '__main__':
    convnext_path = './results/TinyImageNet_CONVNEXT_FP_OW_0/model/final_model-TinyImageNet_CONVNEXT_FP_rep-0.pth'
    resnet50_path = './results/TinyImageNet_RESNET50_FP_OW_1/model/final_model-TinyImageNet_RESNET50_FP_rep-0.pth'

    print("\n" + "#"*100)
    print("# CONVNEXT TINY")
    print("#"*100)
    convnext_data = analyze_model(convnext_path, 'tinyimagenetconvnext')

    print("\n\n" + "#"*100)
    print("# RESNET-50")
    print("#"*100)
    resnet50_data = analyze_model(resnet50_path, 'tinyimagenetresnet50')
