#!/usr/bin/env python3
"""
Verify: BatchNorm (train mode) rescues ResNet-50 from scale mismatch,
but ConvNeXt has no such rescue after its last Linear layer.
"""
import sys, os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.utils.model_compression import get_params_groups_to_quantize, pruning_function_pTTQ_experimental

def load_and_quantize(model_path, model_to_use, alpha=10000.0, init_x=1.0, init_y=1.0):
    if model_to_use == 'tinyimagenetconvnext':
        from src.Models.CNNs.convnext import ConvNeXtTinyClassificationModel
        model_fp = ConvNeXtTinyClassificationModel(num_classes=200)
        model_q = ConvNeXtTinyClassificationModel(num_classes=200)
    else:
        from src.Models.CNNs.resnet50 import ResNet50ClassificationModel
        model_fp = ResNet50ClassificationModel(nb_classes=200)
        model_q = ResNet50ClassificationModel(nb_classes=200)

    data = torch.load(model_path, map_location='cpu', weights_only=False)
    model_fp.load_state_dict(data['model_state_dict'])
    model_q.load_state_dict(data['model_state_dict'])

    # Reset EMA
    if hasattr(pruning_function_pTTQ_experimental, 'ema_states'):
        del pruning_function_pTTQ_experimental.ema_states

    params, names = get_params_groups_to_quantize(model_q, model_to_use)
    weights = params['ToQuantize']['params']
    for i, (name, w) in enumerate(zip(names, weights)):
        pruned = pruning_function_pTTQ_experimental(w.data, alpha, init_x, init_y, k=1.0, beta=0.9, layer_id=i)
        eps = 1e-6
        A = (pruned > eps).float()
        B = (pruned < -eps).float()
        w.data = 1.0 * A + (-1.0 * B)

    return model_fp, model_q


def test_forward(model_to_use, model_fp, model_q):
    x = torch.randn(4, 3, 64, 64)  # batch of 4

    for mode, mode_name in [(True, "TRAIN"), (False, "EVAL")]:
        if mode:
            model_fp.train()
            model_q.train()
        else:
            model_fp.eval()
            model_q.eval()

        with torch.no_grad():
            out_fp = model_fp(x)
            out_q = model_q(x)

        # Check if output is valid (no NaN/Inf)
        fp_valid = torch.isfinite(out_fp).all().item()
        q_valid = torch.isfinite(out_q).all().item()

        if q_valid:
            # Compute top-1 accuracy on this batch (meaningless but shows if model produces sensible output)
            _, pred_fp = out_fp.max(1)
            _, pred_q = out_q.max(1)
            agreement = (pred_fp == pred_q).float().mean().item()

            # Entropy (higher = more uniform = worse)
            probs_fp = out_fp.exp()
            probs_q = out_q.exp()
            entropy_fp = (-probs_fp * out_fp).sum(1).mean().item()
            entropy_q = (-probs_q * out_q).sum(1).mean().item()
            max_prob_fp = probs_fp.max(1)[0].mean().item()
            max_prob_q = probs_q.max(1)[0].mean().item()

            print(f"  {mode_name}: FP entropy={entropy_fp:.3f} maxP={max_prob_fp:.4f} | "
                  f"Q entropy={entropy_q:.3f} maxP={max_prob_q:.4f} | "
                  f"pred agreement={agreement:.2f}")
        else:
            print(f"  {mode_name}: FP valid={fp_valid} | Q valid={q_valid} (overflow/NaN)")

    # Also test: what if we use better initial w_p/w_n?
    print(f"\n  --- With smart initial scales (w_p=w_n computed from FP weights) ---")
    if hasattr(pruning_function_pTTQ_experimental, 'ema_states'):
        del pruning_function_pTTQ_experimental.ema_states

    # Reload quantized model with smart scales
    if model_to_use == 'tinyimagenetconvnext':
        from src.Models.CNNs.convnext import ConvNeXtTinyClassificationModel
        model_smart = ConvNeXtTinyClassificationModel(num_classes=200)
    else:
        from src.Models.CNNs.resnet50 import ResNet50ClassificationModel
        model_smart = ResNet50ClassificationModel(nb_classes=200)
    model_smart.load_state_dict(model_fp.state_dict())

    params, names = get_params_groups_to_quantize(model_smart, model_to_use)
    weights = params['ToQuantize']['params']
    for i, (name, w) in enumerate(zip(names, weights)):
        fp_data = w.data.clone()
        pruned = pruning_function_pTTQ_experimental(fp_data, 10000.0, 1.0, 1.0, k=1.0, beta=0.9, layer_id=i)
        eps = 1e-6
        A = (pruned > eps).float()
        B = (pruned < -eps).float()

        # Smart scales: mean magnitude of positive/negative FP weights above threshold
        pos_vals = fp_data[A.bool()]
        neg_vals = (-fp_data[B.bool()])
        w_p = pos_vals.mean().item() if len(pos_vals) > 0 else 1.0
        w_n = neg_vals.mean().item() if len(neg_vals) > 0 else 1.0

        w.data = w_p * A + (-w_n * B)

    model_smart.eval()
    with torch.no_grad():
        out_smart = model_smart(x)
    if torch.isfinite(out_smart).all():
        probs_smart = out_smart.exp()
        entropy_smart = (-probs_smart * out_smart).sum(1).mean().item()
        max_prob_smart = probs_smart.max(1)[0].mean().item()
        _, pred_smart = out_smart.max(1)
        _, pred_fp_eval = model_fp(x).max(1)
        agreement = (pred_fp_eval == pred_smart).float().mean().item()
        print(f"  EVAL+smart: entropy={entropy_smart:.3f} maxP={max_prob_smart:.4f} | "
              f"pred agreement with FP={agreement:.2f}")
    else:
        print(f"  EVAL+smart: output invalid (overflow/NaN)")

    # Also test: what about weight normalization?
    print(f"\n  --- With weight normalization (do_normalization_weights=true) ---")
    if hasattr(pruning_function_pTTQ_experimental, 'ema_states'):
        del pruning_function_pTTQ_experimental.ema_states

    if model_to_use == 'tinyimagenetconvnext':
        from src.Models.CNNs.convnext import ConvNeXtTinyClassificationModel
        model_norm = ConvNeXtTinyClassificationModel(num_classes=200)
    else:
        from src.Models.CNNs.resnet50 import ResNet50ClassificationModel
        model_norm = ResNet50ClassificationModel(nb_classes=200)
    model_norm.load_state_dict(model_fp.state_dict())

    params, names = get_params_groups_to_quantize(model_norm, model_to_use)
    weights = params['ToQuantize']['params']

    # Normalize weights first
    for name, w in zip(names, weights):
        w.data = w.data / w.data.abs().max()

    # Then quantize with w_p=w_n=1.0
    for i, (name, w) in enumerate(zip(names, weights)):
        pruned = pruning_function_pTTQ_experimental(w.data, 10000.0, 1.0, 1.0, k=1.0, beta=0.9, layer_id=i)
        eps = 1e-6
        A = (pruned > eps).float()
        B = (pruned < -eps).float()
        w.data = 1.0 * A + (-1.0 * B)

    model_norm.eval()
    with torch.no_grad():
        out_norm = model_norm(x)
    if torch.isfinite(out_norm).all():
        probs_norm = out_norm.exp()
        entropy_norm = (-probs_norm * out_norm).sum(1).mean().item()
        max_prob_norm = probs_norm.max(1)[0].mean().item()
        print(f"  EVAL+norm:  entropy={entropy_norm:.3f} maxP={max_prob_norm:.4f}")
    else:
        print(f"  EVAL+norm:  output invalid (overflow/NaN)")


if __name__ == '__main__':
    convnext_path = './results/TinyImageNet_CONVNEXT_FP_OW_0/model/final_model-TinyImageNet_CONVNEXT_FP_rep-0.pth'
    resnet50_path = './results/TinyImageNet_RESNET50_FP_OW_1/model/final_model-TinyImageNet_RESNET50_FP_rep-0.pth'

    for model_path, model_to_use in [
        (convnext_path, 'tinyimagenetconvnext'),
        (resnet50_path, 'tinyimagenetresnet50'),
    ]:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_to_use}")
        print(f"{'='*80}")
        model_fp, model_q = load_and_quantize(model_path, model_to_use)
        test_forward(model_to_use, model_fp, model_q)
