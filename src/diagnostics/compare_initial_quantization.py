#!/usr/bin/env python3
"""
Compare the effect of initial quantization on ConvNeXt vs ResNet-50.
Key question: Why does EMA-pTTQ initial quantization destroy ConvNeXt but not ResNet-50?
"""
import sys, os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.utils.model_compression import get_params_groups_to_quantize, pruning_function_pTTQ_experimental

def load_model(model_path, model_to_use):
    if model_to_use == 'tinyimagenetconvnext':
        from src.Models.CNNs.convnext import ConvNeXtTinyClassificationModel
        model = ConvNeXtTinyClassificationModel(num_classes=200)
    elif model_to_use == 'tinyimagenetresnet50':
        from src.Models.CNNs.resnet50 import ResNet50ClassificationModel
        model = ResNet50ClassificationModel(nb_classes=200)
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(data['model_state_dict'])
    return model

def quantize_initial(model, model_to_use, alpha=10000.0, init_x=1.0, init_y=1.0, k=1.0, beta=0.9):
    """Apply initial EMA-pTTQ quantization with w_p=w_n=1.0 and measure the effect."""
    params, names = get_params_groups_to_quantize(model, model_to_use)
    weights = params['ToQuantize']['params']

    # Reset EMA states
    if hasattr(pruning_function_pTTQ_experimental, 'ema_states'):
        del pruning_function_pTTQ_experimental.ema_states

    print(f"\n{'='*80}")
    print(f"Initial Quantization Analysis: {model_to_use}")
    print(f"{'='*80}")

    for i, (name, w) in enumerate(zip(names, weights)):
        fp_data = w.data.clone()

        # Apply pruning function (same as quantize() in experiment_pTTQ_experimental.py)
        pruned = pruning_function_pTTQ_experimental(fp_data, alpha, init_x, init_y, k=k, beta=beta, layer_id=i)
        eps = 1e-6
        A = (pruned > eps).float()
        B = (pruned < -eps).float()

        # Quantize with w_p=w_n=1.0
        w_p, w_n = 1.0, 1.0
        quantized = w_p * A + (-w_n * B)

        # Compute statistics
        fp_std = fp_data.std().item()
        fp_abs_mean = fp_data.abs().mean().item()
        quant_abs_mean = quantized.abs().mean().item()
        quant_nonzero_mean = quantized[quantized != 0].abs().mean().item() if (quantized != 0).any() else 0

        sparsity = (quantized == 0).float().mean().item()

        # Scale ratio: how much the output magnitude changes
        # For a matrix multiply Y = X @ W, the output scale changes by scale_ratio
        fp_rms = (fp_data ** 2).mean().sqrt().item()
        quant_rms = (quantized ** 2).mean().sqrt().item()
        scale_ratio = quant_rms / fp_rms if fp_rms > 0 else float('inf')

        # Cosine similarity between FP and quantized (measures information preservation)
        fp_flat = fp_data.flatten().float()
        quant_flat = quantized.flatten().float()
        cos_sim = torch.nn.functional.cosine_similarity(fp_flat.unsqueeze(0), quant_flat.unsqueeze(0)).item()

        # Relative quantization error
        quant_error = (fp_data - quantized).norm() / fp_data.norm()

        # Determine layer type
        if w.data.ndim == 4 and w.data.shape[1] == 1:
            ltype = "DW"
        elif w.data.ndim == 2:
            ltype = "FC"
        elif w.data.ndim == 4:
            ltype = f"C{w.data.shape[2]}x{w.data.shape[3]}"
        else:
            ltype = "??"

        print(f"  [{ltype:<5s}] {name:<55s} sp={sparsity:.0%} | FP_rms={fp_rms:.4f} Q_rms={quant_rms:.4f} scale={scale_ratio:.1f}x | cos={cos_sim:.3f} qerr={quant_error:.3f}")

        # Apply quantization to the model
        w.data = quantized

    return model

def forward_pass_comparison(model_to_use, model_fp, model_quant):
    """Compare forward pass outputs."""
    # Create a dummy input
    if 'convnext' in model_to_use:
        x = torch.randn(1, 3, 64, 64)
    else:
        x = torch.randn(1, 3, 64, 64)

    model_fp.eval()
    model_quant.eval()

    with torch.no_grad():
        out_fp = model_fp(x)
        out_quant = model_quant(x)

    print(f"\n  Forward pass comparison:")
    print(f"    FP output:   mean={out_fp.mean():.4f} std={out_fp.std():.4f} max={out_fp.max():.4f}")
    print(f"    Quant output: mean={out_quant.mean():.4f} std={out_quant.std():.4f} max={out_quant.max():.4f}")
    print(f"    FP is log_softmax, so checking exp:")
    print(f"    FP probs:   max={out_fp.exp().max():.4f} entropy={(-out_fp.exp() * out_fp).sum():.4f}")
    print(f"    Quant probs: max={out_quant.exp().max():.4f} entropy={(-out_quant.exp() * out_quant).sum():.4f}")

    # Check if output is near uniform (random)
    uniform_log = -np.log(200)  # ln(1/200) for 200 classes
    print(f"    Uniform distribution: log_softmax would be ~{uniform_log:.4f}")
    print(f"    FP deviation from uniform: {(out_fp - uniform_log).abs().mean():.4f}")
    print(f"    Quant deviation from uniform: {(out_quant - uniform_log).abs().mean():.4f}")

if __name__ == '__main__':
    convnext_path = './results/TinyImageNet_CONVNEXT_FP_OW_0/model/final_model-TinyImageNet_CONVNEXT_FP_rep-0.pth'
    resnet50_path = './results/TinyImageNet_RESNET50_FP_OW_1/model/final_model-TinyImageNet_RESNET50_FP_rep-0.pth'

    for model_path, model_to_use in [
        (convnext_path, 'tinyimagenetconvnext'),
        (resnet50_path, 'tinyimagenetresnet50'),
    ]:
        # Load fresh models
        model_fp = load_model(model_path, model_to_use)
        model_quant = load_model(model_path, model_to_use)

        # Quantize one copy
        model_quant = quantize_initial(model_quant, model_to_use)

        # Compare forward passes
        forward_pass_comparison(model_to_use, model_fp, model_quant)
