#!/usr/bin/env python3
"""Load and inspect training results from EMA-pTTQ runs."""
import pickle
import sys
import numpy as np
import glob

def deep_inspect(obj, prefix="", depth=0):
    if depth > 5:
        return
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=str):
            val = obj[key]
            deep_inspect(val, prefix=f"{prefix}.{key}", depth=depth+1)
    elif isinstance(obj, (list, tuple)):
        arr = np.array(obj) if len(obj) > 0 else np.array([])
        if arr.ndim == 1 and len(arr) > 0 and np.issubdtype(arr.dtype, np.number):
            print(f"  {prefix}: len={len(arr)}, last={arr[-1]:.6f}, max={arr.max():.6f}@idx{arr.argmax()}, min={arr.min():.6f}")
        else:
            print(f"  {prefix}: list/tuple len={len(obj)}")
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            print(f"  {prefix}: scalar={obj.item():.6f}")
        elif obj.ndim == 1 and len(obj) > 0:
            print(f"  {prefix}: len={len(obj)}, last={obj[-1]:.6f}, max={obj.max():.6f}@idx{obj.argmax()}, min={obj.min():.6f}")
        else:
            print(f"  {prefix}: ndarray shape={obj.shape}")
    elif isinstance(obj, (float, int, np.floating, np.integer)):
        print(f"  {prefix}: {obj}")
    else:
        print(f"  {prefix}: type={type(obj).__name__}")

def inspect_results(path, label=""):
    with open(path, "rb") as f:
        results = pickle.load(f)
    print(f"\n{'='*80}")
    print(f"Results: {label}")
    print(f"{'='*80}")
    deep_inspect(results, prefix="root")

if __name__ == '__main__':
    paths = [
        ("./results/TinyImageNet_CONVNEXT_EMA_PTTQ_k1_beta0.9_OW_0/metrics/final_results_all_repetitions.pth", "ConvNeXt EMA-pTTQ"),
    ]
    for d in glob.glob("./results/TinyImageNet_RESNET50*EMA*PTTQ*OW_*/metrics/final_results_all_repetitions.pth"):
        paths.append((d, "ResNet-50 EMA-pTTQ"))
    for d in glob.glob("./results/TinyImageNet_CONVNEXT_TTQ_OW_*/metrics/final_results_all_repetitions.pth"):
        paths.append((d, "ConvNeXt TTQ"))

    for path, label in paths:
        try:
            inspect_results(path, label)
        except Exception as e:
            print(f"Error loading {path}: {e}")
