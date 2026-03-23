#!/usr/bin/env python
"""
BERT Ternary Quantization on GLUE Tasks.

Supports: FP baseline, TTQ, pTTQ, EMA-pTTQ.
Tasks: CoLA, MRPC, RTE, STS-B, SST-2, QNLI.
Runs median-of-5 (seeds 41-45) and saves results to ./results/glue_ternary.csv.

Usage:
    source env/bin/activate
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export TORCH_HOME=./data HF_HOME=./data

    python src/Experiments/train_glue_quantized.py --task_name cola --method fp
    python src/Experiments/train_glue_quantized.py --task_name cola --method ttq --smart_initial_scales
    python src/Experiments/train_glue_quantized.py --task_name cola --method pttq --smart_initial_scales
    python src/Experiments/train_glue_quantized.py --task_name cola --method ema_pttq --smart_initial_scales
"""

import argparse
import csv
import fcntl
import gc
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import datasets
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

# Import pruning functions from existing codebase
from src.utils.model_compression import (
    pruning_function_pTTQ,
    pruning_function_pTTQ_experimental,
)

###############################################################################
#                              Constants                                      #
###############################################################################
SEEDS: List[int] = [41, 42, 43, 44, 45]
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "glue_ternary.csv")

_METRIC_FOR_TASK = {
    "cola": "matthews_correlation",
    "mrpc": "f1",
    "rte": "accuracy",
    "stsb": "spearmanr",
    "sst2": "accuracy",
    "qnli": "accuracy",
}

task_to_keys = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
}

logger = logging.getLogger(__name__)

###############################################################################
#                              Helpers                                        #
###############################################################################
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def primary_metric(task_name: str, metric_dict: dict) -> float:
    key = _METRIC_FOR_TASK.get(task_name, "accuracy")
    return metric_dict.get(key, float("-inf"))


###############################################################################
#                        Layer Identification                                 #
###############################################################################
def get_layers_to_quantize(model) -> List[Tuple[str, nn.Parameter]]:
    """Identify all Linear weight parameters in BERT encoder to quantize.

    Quantizes: encoder.layer.*.{attention,intermediate,output}.*.weight
    Skips: embeddings, LayerNorm, classifier head, all biases.
    """
    layers = []
    for name, param in model.named_parameters():
        if ("encoder.layer" in name
                and name.endswith(".weight")
                and "LayerNorm" not in name):
            layers.append((name, param))
    return layers


###############################################################################
#                     Smart Initial Scales                                    #
###############################################################################
def compute_smart_scales(kernel: torch.Tensor, init_x: float) -> Tuple[float, float]:
    """Compute initial w_p, w_n from FP weight statistics.

    Matches the quantized output scale to the FP output scale.
    Critical for architectures without BatchNorm after quantized layers (BERT, ConvNeXt).
    """
    k_mean, k_std = kernel.mean(), kernel.std()
    delta = abs(k_mean + init_x * k_std)

    pos_mask = kernel > delta
    neg_mask = kernel < -delta

    w_p = kernel[pos_mask].mean().item() if pos_mask.any() else 1.0
    w_n = (-kernel[neg_mask]).mean().item() if neg_mask.any() else 1.0
    return w_p, w_n


###############################################################################
#                        Quantization Functions                               #
###############################################################################
def quantize_ttq(kernel, w_p, w_n, t=0.05):
    """TTQ: fixed threshold at t * max(|kernel|)."""
    delta = t * kernel.abs().max()
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    return w_p * a + (-w_n * b)


def quantize_pttq(kernel, w_p, w_n, alpha, a_thresh, b_thresh, k=1.0):
    """pTTQ: stats-based learnable thresholds, with optional k scaling."""
    pruned = pruning_function_pTTQ(kernel, alpha, k * a_thresh, k * b_thresh)
    eps = 1e-6
    A = (pruned > eps).float()
    B = (pruned < -eps).float()
    return w_p * A + (-w_n * B)


def quantize_ema_pttq(kernel, w_p, w_n, alpha, a_thresh, b_thresh,
                       k=1.0, beta=0.9, layer_id=0):
    """EMA-pTTQ: EMA-smoothed stats-based thresholds."""
    pruned = pruning_function_pTTQ_experimental(
        kernel, alpha, a_thresh, b_thresh, k, beta, layer_id)
    eps = 1e-6
    A = (pruned > eps).float()
    B = (pruned < -eps).float()
    return w_p * A + (-w_n * B)


###############################################################################
#                         Gradient Functions (STE)                            #
###############################################################################
def get_grads_ttq(kernel_grad, kernel, w_p, w_n, t, device):
    """Compute STE gradients for TTQ.

    Returns: (grad_fp, grad_wp, grad_wn)
    """
    delta = t * kernel.abs().max()
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    c = torch.ones_like(kernel) - a - b

    grad_fp = w_p * a * kernel_grad + w_n * b * kernel_grad + 1.0 * c * kernel_grad
    grad_wp = (a * kernel_grad).sum()
    grad_wn = (b * kernel_grad).sum()
    return grad_fp, grad_wp, grad_wn


def _compute_threshold_grads(kernel_grad, kernel, alpha, a_thresh, b_thresh,
                              device, k=1.0):
    """Shared threshold gradient computation for pTTQ and EMA-pTTQ (stats-based).

    Both methods use delta = k * |mean + param * std| thresholds, so the gradient
    formulas are identical. Only the forward-pass threshold smoothing differs.
    When k != 1.0, deltas are k-scaled and grad_a/grad_b include an extra k
    factor from the chain rule: d(k*a)/d(a) = k.

    Returns: (grad_a, grad_b, grad_alpha)
    """
    kernel_mean, kernel_std = kernel.mean(), kernel.std()
    delta_min = k * (kernel_mean + a_thresh * kernel_std).abs()
    delta_max = k * (kernel_mean + b_thresh * kernel_std).abs()

    zero_t = torch.tensor([0.0], device=device)
    sigmoid = torch.nn.functional.sigmoid

    # grad_a: dL/da via STE, loss-modulated by kernel_grad
    sig_neg = sigmoid(alpha * (-kernel - delta_min))
    grad_a = k * (kernel_grad * (
        kernel_std * torch.heaviside((-kernel - delta_min).float(), zero_t)
        - kernel_std * sig_neg
        + kernel_std * delta_min * alpha * sig_neg * (1 - sig_neg)
    )).sum()

    # grad_b: dL/db via STE
    sig_pos = sigmoid(alpha * (kernel - delta_max))
    grad_b = k * (kernel_grad * (
        -kernel_std * torch.heaviside((kernel - delta_max).float(), zero_t)
        + kernel_std * sig_pos
        - kernel_std * delta_max * alpha * sig_pos * (1 - sig_pos)
    )).sum()

    # grad_alpha: dL/d(alpha) via STE (uses k-scaled deltas, no extra k factor)
    grad_alpha = (kernel_grad * (
        delta_max * (kernel - delta_max) * sig_pos * (1 - sig_pos)
        + delta_min * (kernel + delta_min) * sig_neg * (1 - sig_neg)
    )).sum()

    return grad_a, grad_b, grad_alpha


def get_grads_pttq(kernel_grad, kernel, w_p, w_n, alpha, a_thresh, b_thresh,
                    device, k=1.0):
    """Compute STE gradients for pTTQ (stats-based thresholds, with k scaling).

    Returns: (grad_fp, grad_wp, grad_wn, grad_a, grad_b, grad_alpha)
    """
    pruned = pruning_function_pTTQ(kernel, alpha, k * a_thresh, k * b_thresh)
    eps = 1e-6
    A = (pruned > eps).float()
    B = (pruned < -eps).float()
    c = torch.ones_like(pruned) - A - B

    grad_fp = w_p * A * kernel_grad + w_n * B * kernel_grad + 1.0 * c * kernel_grad
    grad_wp = (A * kernel_grad).sum()
    grad_wn = (B * kernel_grad).sum()

    grad_a, grad_b, grad_alpha = _compute_threshold_grads(
        kernel_grad, kernel, alpha, a_thresh, b_thresh, device, k=k)

    return grad_fp, grad_wp, grad_wn, grad_a, grad_b, grad_alpha


def get_grads_ema_pttq(kernel_grad, kernel, w_p, w_n, alpha, a_thresh, b_thresh,
                        k, beta, layer_id, device):
    """Compute STE gradients for EMA-pTTQ.

    Note: The gradient formula is identical to pTTQ (stats-based). The EMA only
    affects the forward-pass pruning function, not the analytical gradient.
    Calling the pruning function here also updates the EMA state (matches
    existing code behavior where EMA updates twice per step).

    Returns: (grad_fp, grad_wp, grad_wn, grad_a, grad_b, grad_alpha)
    """
    pruned = pruning_function_pTTQ_experimental(
        kernel, alpha, a_thresh, b_thresh, k, beta, layer_id)
    eps = 1e-6
    A = (pruned > eps).float()
    B = (pruned < -eps).float()
    c = torch.ones_like(pruned) - A - B

    grad_fp = w_p * A * kernel_grad + w_n * B * kernel_grad + 1.0 * c * kernel_grad
    grad_wp = (A * kernel_grad).sum()
    grad_wn = (B * kernel_grad).sum()

    grad_a, grad_b, grad_alpha = _compute_threshold_grads(
        kernel_grad, kernel, alpha, a_thresh, b_thresh, device, k=k)

    return grad_fp, grad_wp, grad_wn, grad_a, grad_b, grad_alpha


###############################################################################
#                          Sparsity & Stats                                   #
###############################################################################
def compute_sparsity(model, layer_names: List[str]) -> float:
    """Compute overall sparsity (fraction of zeros) across quantized layers."""
    total = 0
    zeros = 0
    for name, param in model.named_parameters():
        if name in layer_names:
            total += param.numel()
            zeros += (param.data == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def compute_compression_stats(model, layer_names: List[str]) -> Dict:
    """Compute compression statistics for quantized model."""
    total_params = sum(p.numel() for p in model.parameters())
    quant_params = sum(p.numel() for n, p in model.named_parameters() if n in layer_names)
    non_quant_params = total_params - quant_params

    # Ternary: 2 bits/weight, FP32: 32 bits/weight
    fp_bits = total_params * 32
    quant_bits = quant_params * 2 + non_quant_params * 32
    compression_ratio = fp_bits / quant_bits if quant_bits > 0 else 1.0

    return {
        "total_params": total_params,
        "quant_params": quant_params,
        "quant_fraction": quant_params / total_params,
        "compression_ratio": compression_ratio,
    }


def compute_per_layer_sparsity(model, layer_names: List[str]) -> Dict[str, float]:
    """Compute sparsity (fraction of zeros) for each quantized layer."""
    per_layer = {}
    for name, param in model.named_parameters():
        if name in layer_names:
            total = param.numel()
            zeros = (param.data == 0).sum().item()
            per_layer[name] = zeros / total if total > 0 else 0.0
    return per_layer


###############################################################################
#                     Quantized Optimization Step                             #
###############################################################################
def quantized_optimize_step(
    method, quant_params, fp_copies, scaling_factors, thresholds,
    optimizer_model, optimizer_fp, optimizer_sf, optimizer_t,
    scheduler, alpha, t_ttq, k_ema, beta_ema, device, step_counter,
    layer_names,
):
    """Perform the STE-based optimization step for quantized training.

    1. Compute STE gradients for each quantized layer
    2. Step all optimizers
    3. Clamp thresholds to positive
    4. Re-quantize all layers

    Must be called AFTER loss.backward().
    """
    n_layers = len(quant_params)

    # --- Phase 1: Compute gradients ---
    for i in range(n_layers):
        k = quant_params[i]      # quantized weight (in model)
        k_fp = fp_copies[i]      # FP copy
        sf = scaling_factors[i]  # [w_p, w_n]
        w_p, w_n = sf.data[0], sf.data[1]

        if k.grad is None:
            continue

        if method == "ttq":
            k_fp_grad, wp_grad, wn_grad = get_grads_ttq(
                k.grad.data, k_fp.data, w_p, w_n, t_ttq, device)
            k_fp.grad = Variable(k_fp_grad)
            k.grad.data.zero_()
            sf.grad = Variable(torch.FloatTensor([wp_grad, wn_grad]).to(device))

        elif method == "pttq":
            th = thresholds[i]
            a_val, b_val = th.data[0], th.data[1]
            k_fp_grad, wp_grad, wn_grad, a_grad, b_grad, _ = get_grads_pttq(
                k.grad.data, k_fp.data, w_p, w_n, alpha, a_val, b_val, device,
                k=k_ema)
            k_fp.grad = Variable(k_fp_grad)
            k.grad.data.zero_()
            sf.grad = Variable(torch.FloatTensor([wp_grad, wn_grad]).to(device))
            th.grad = Variable(torch.FloatTensor([a_grad, b_grad]).to(device))

        elif method == "ema_pttq":
            th = thresholds[i]
            a_val, b_val = th.data[0], th.data[1]
            k_fp_grad, wp_grad, wn_grad, a_grad, b_grad, _ = get_grads_ema_pttq(
                k.grad.data, k_fp.data, w_p, w_n, alpha,
                a_val, b_val, k_ema, beta_ema, i, device)
            k_fp.grad = Variable(k_fp_grad)
            k.grad.data.zero_()
            sf.grad = Variable(torch.FloatTensor([wp_grad, wn_grad]).to(device))
            th.grad = Variable(torch.FloatTensor([a_grad, b_grad]).to(device))

    # --- Phase 2: Step all optimizers ---
    optimizer_model.step()
    optimizer_fp.step()
    optimizer_sf.step()
    if optimizer_t is not None:
        optimizer_t.step()
    scheduler.step()

    # --- Phase 3: Clamp thresholds to positive ---
    if thresholds:
        for th in thresholds:
            th.data[0] = torch.clamp(th.data[0], min=1e-8)
            th.data[1] = torch.clamp(th.data[1], min=1e-8)

    # --- Phase 4: Re-quantize with updated params ---
    for i in range(n_layers):
        k = quant_params[i]
        k_fp = fp_copies[i]
        sf = scaling_factors[i]
        w_p, w_n = sf.data[0], sf.data[1]

        with torch.no_grad():
            if method == "ttq":
                k.data = quantize_ttq(k_fp.data, w_p, w_n, t_ttq)
            elif method == "pttq":
                th = thresholds[i]
                k.data = quantize_pttq(
                    k_fp.data, w_p, w_n, alpha, th.data[0], th.data[1],
                    k=k_ema)
            elif method == "ema_pttq":
                th = thresholds[i]
                k.data = quantize_ema_pttq(
                    k_fp.data, w_p, w_n, alpha,
                    th.data[0], th.data[1], k_ema, beta_ema, i)

    # --- Diagnostic logging every 100 steps ---
    if step_counter % 100 == 1:
        print(f"\n--- DIAG step {step_counter} ---")
        for i in range(min(n_layers, 6)):  # Print first 6 layers
            k = quant_params[i]
            k_fp = fp_copies[i]
            sf = scaling_factors[i]
            total = k.data.numel()
            zero_ct = (k.data == 0).sum().item()
            name = layer_names[i] if i < len(layer_names) else f"layer_{i}"
            line = (f"  [{name}] wp={sf.data[0]:.4f} wn={sf.data[1]:.4f} | "
                    f"FP range=[{k_fp.data.min():.4f},{k_fp.data.max():.4f}] "
                    f"std={k_fp.data.std():.4f} | sparsity={zero_ct/total:.1%}")
            if thresholds:
                th = thresholds[i]
                line += f" | a={th.data[0]:.6f} b={th.data[1]:.6f}"
                if th.grad is not None:
                    line += f" | grad_a={th.grad.data[0]:.6f} grad_b={th.grad.data[1]:.6f}"
            print(line)
        if n_layers > 6:
            print(f"  ... ({n_layers - 6} more layers)")
        print("--- end DIAG ---\n")


###############################################################################
#                           Run Single Seed                                   #
###############################################################################
def run_single_seed(args, seed: int) -> Dict:
    set_seed(seed)
    device = torch.device(args.device)
    method = args.method

    logger.info(f"[seed {seed}] {method.upper()} on {args.task_name} | device={device}")

    # --- Data Loading ---
    raw_datasets = load_dataset("glue", args.task_name)
    is_regression = (args.task_name == "stsb")
    if is_regression:
        num_labels = 1
    else:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)

    # --- Model ---
    config = AutoConfig.from_pretrained(
        args.model_name, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config)
    model.to(device)

    # --- Tokenization ---
    s1_key, s2_key = task_to_keys[args.task_name]

    def preprocess(examples):
        texts = ((examples[s1_key],) if s2_key is None
                 else (examples[s1_key], examples[s2_key]))
        result = tokenizer(*texts, padding="max_length",
                           max_length=args.max_length, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    processed = raw_datasets.map(
        preprocess, batched=True,
        remove_columns=raw_datasets["train"].column_names, desc="Tokenizing")
    train_dataset = processed["train"]
    eval_dataset = processed["validation"]

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=args.batch_size, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             collate_fn=collator)

    # --- Warmup + Quantization Setup ---
    warmup_epochs = args.warmup_epochs if method != "fp" else 0
    total_epochs = warmup_epochs + args.epochs
    total_steps = len(train_loader) * total_epochs
    lr_warmup_steps = int(0.1 * total_steps)

    quant_layers = []       # (name, param) pairs
    layer_names = []        # names only
    fp_copies = []          # FP weight copies
    scaling_factors = []    # [w_p, w_n] per layer
    thresholds = []         # [a, b] per layer (pTTQ/EMA-pTTQ only)
    quant_ids = set()
    quantization_active = False

    if method != "fp":
        # Reset EMA state between seeds
        if hasattr(pruning_function_pTTQ_experimental, "ema_states"):
            del pruning_function_pTTQ_experimental.ema_states

        # Identify layers (but don't quantize yet if warmup)
        quant_layers = get_layers_to_quantize(model)
        layer_names = [name for name, _ in quant_layers]
        quant_ids = {id(p) for _, p in quant_layers}
        logger.info(f"[seed {seed}] Will quantize {len(layer_names)} layers"
                    f" (warmup={warmup_epochs} epochs)")

    # --- Optimizers (start with all-params optimizer for warmup) ---
    # For FP or warmup phase: optimize all model params
    optimizer_model = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        args.scheduler_type, optimizer=optimizer_model,
        num_warmup_steps=lr_warmup_steps, num_training_steps=total_steps)
    optimizer_fp = optimizer_sf = optimizer_t = None

    # --- Helper: Apply quantization ---
    def apply_quantization():
        nonlocal fp_copies, scaling_factors, thresholds
        nonlocal optimizer_model, optimizer_fp, optimizer_sf, optimizer_t, scheduler

        fp_copies.clear()
        scaling_factors.clear()
        thresholds.clear()

        for i, (name, param) in enumerate(quant_layers):
            fp_copy = Variable(param.data.clone(), requires_grad=True)
            fp_copies.append(fp_copy)

            if args.smart_initial_scales:
                w_p, w_n = compute_smart_scales(param.data, args.init_x)
            else:
                w_p, w_n = 1.0, 1.0
            sf = Variable(torch.FloatTensor([w_p, w_n]).to(device),
                          requires_grad=True)
            scaling_factors.append(sf)

            if method in ("pttq", "ema_pttq"):
                th = Variable(torch.FloatTensor([args.init_x, args.init_y]).to(device),
                              requires_grad=True)
                thresholds.append(th)

            with torch.no_grad():
                if method == "ttq":
                    param.data = quantize_ttq(fp_copy.data, w_p, w_n, args.t)
                elif method == "pttq":
                    param.data = quantize_pttq(
                        fp_copy.data, w_p, w_n, args.alpha,
                        args.init_x, args.init_y, k=args.k)
                elif method == "ema_pttq":
                    param.data = quantize_ema_pttq(
                        fp_copy.data, w_p, w_n, args.alpha,
                        args.init_x, args.init_y, args.k, args.beta, i)

        init_sparsity = compute_sparsity(model, layer_names)
        comp_stats = compute_compression_stats(model, layer_names)
        thresh_lr = args.lr_thresh if args.lr_thresh is not None else args.lr_quant
        logger.info(f"[seed {seed}] Quantization applied! "
                    f"Sparsity: {init_sparsity:.1%} | "
                    f"{comp_stats['quant_params']:,}/{comp_stats['total_params']:,} "
                    f"params ({comp_stats['quant_fraction']:.1%}) | "
                    f"Compression: {comp_stats['compression_ratio']:.1f}x | "
                    f"lr_sf={args.lr_quant} lr_thresh={thresh_lr} "
                    f"opt_thresh={args.optimizer_thresh}")

        # Switch optimizer_model to non-quantized params only
        model_params = [p for p in model.parameters()
                        if id(p) not in quant_ids and p.requires_grad]
        optimizer_model = torch.optim.AdamW(
            model_params, lr=args.lr, weight_decay=args.weight_decay)
        # Create new scheduler for the quantized phase
        quant_steps = len(train_loader) * args.epochs
        scheduler = get_scheduler(
            args.scheduler_type, optimizer=optimizer_model,
            num_warmup_steps=int(0.1 * quant_steps),
            num_training_steps=quant_steps)

        optimizer_fp = torch.optim.Adamax(fp_copies, lr=args.lr)
        optimizer_sf = torch.optim.Adamax(scaling_factors, lr=args.lr_quant)
        optimizer_t = None
        if method in ("pttq", "ema_pttq"):
            thresh_lr = args.lr_thresh if args.lr_thresh is not None else args.lr_quant
            if args.optimizer_thresh == "sgd":
                optimizer_t = torch.optim.SGD(thresholds, lr=thresh_lr)
            else:
                optimizer_t = torch.optim.Adamax(thresholds, lr=thresh_lr)

    # If no warmup, apply quantization immediately
    if method != "fp" and warmup_epochs == 0:
        apply_quantization()
        quantization_active = True

    # --- Metric ---
    metric = evaluate.load("glue", args.task_name)

    # --- Training Loop ---
    best_metric_val = float("-inf")
    best_metric_dict = {}
    best_sparsity = 0.0
    best_per_layer_sparsity = {}
    global_step = 0

    for epoch in range(total_epochs):
        # Transition from warmup to quantized training
        if (method != "fp" and not quantization_active
                and epoch == warmup_epochs):
            logger.info(f"[seed {seed}] Warmup complete. Applying quantization...")
            apply_quantization()
            quantization_active = True
            # Reset best-metric tracking so we only track quantized-phase results
            best_metric_val = float("-inf")
            best_metric_dict = {}
            best_sparsity = 0.0
            best_per_layer_sparsity = {}

        is_quant_step = (method != "fp" and quantization_active)

        model.train()
        epoch_loss = 0.0

        phase = "quant" if is_quant_step else "warmup" if warmup_epochs > 0 else "fp"
        progress = tqdm(train_loader,
                        desc=f"[seed {seed}] Epoch {epoch+1}/{total_epochs} ({phase})")
        for step, batch in enumerate(progress):
            batch = {bk: bv.to(device, non_blocking=True) for bk, bv in batch.items()}
            global_step += 1

            if not is_quant_step:
                # --- Standard fine-tuning (FP or warmup) ---
                optimizer_model.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_model.step()
                scheduler.step()
            else:
                # --- Quantized training with STE ---
                optimizer_model.zero_grad()
                optimizer_fp.zero_grad()
                optimizer_sf.zero_grad()
                if optimizer_t is not None:
                    optimizer_t.zero_grad()

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                # Clip gradients for non-quantized params
                non_quant_params = [p for p in model.parameters()
                                    if id(p) not in quant_ids and p.requires_grad]
                torch.nn.utils.clip_grad_norm_(non_quant_params, 1.0)

                quant_params_list = [p for _, p in quant_layers]
                quantized_optimize_step(
                    method=method,
                    quant_params=quant_params_list,
                    fp_copies=fp_copies,
                    scaling_factors=scaling_factors,
                    thresholds=thresholds,
                    optimizer_model=optimizer_model,
                    optimizer_fp=optimizer_fp,
                    optimizer_sf=optimizer_sf,
                    optimizer_t=optimizer_t,
                    scheduler=scheduler,
                    alpha=args.alpha,
                    t_ttq=args.t,
                    k_ema=args.k,
                    beta_ema=args.beta,
                    device=device,
                    step_counter=global_step,
                    layer_names=layer_names,
                )

            epoch_loss += loss.item()
            if step % 50 == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        # --- Evaluation ---
        model.eval()
        all_preds = []
        all_refs = []
        for batch in eval_loader:
            batch = {bk: bv.to(device, non_blocking=True) for bk, bv in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            if is_regression:
                preds = outputs.logits.squeeze(-1)
            else:
                preds = outputs.logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_refs.append(batch["labels"].cpu())
            metric.add_batch(predictions=preds.cpu(),
                             references=batch["labels"].cpu())

        eval_metric = metric.compute()
        sparsity = compute_sparsity(model, layer_names) if method != "fp" else 0.0
        prim = primary_metric(args.task_name, eval_metric)

        # Log prediction distribution
        all_preds_t = torch.cat(all_preds)
        all_refs_t = torch.cat(all_refs)
        if is_regression:
            logger.info(f"[seed {seed}] Epoch {epoch+1}: {eval_metric} | "
                        f"sparsity={sparsity:.1%} | loss={avg_loss:.4f} | "
                        f"pred_mean={all_preds_t.mean():.3f} pred_std={all_preds_t.std():.3f}")
        else:
            pred_counts = {i: (all_preds_t == i).sum().item() for i in range(num_labels)}
            ref_counts = {i: (all_refs_t == i).sum().item() for i in range(num_labels)}
            logger.info(f"[seed {seed}] Epoch {epoch+1}: {eval_metric} | "
                        f"sparsity={sparsity:.1%} | loss={avg_loss:.4f} | "
                        f"preds={pred_counts} refs={ref_counts}")

        if prim > best_metric_val:
            best_metric_val = prim
            best_metric_dict = eval_metric.copy()
            best_sparsity = sparsity
            best_per_layer_sparsity = (
                compute_per_layer_sparsity(model, layer_names) if method != "fp" else {}
            )

    # Compute final compression stats
    comp_stats = compute_compression_stats(model, layer_names) if method != "fp" else {}
    peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # --- Cleanup ---
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "best_metric_dict": best_metric_dict,
        "sparsity": best_sparsity,
        "per_layer_sparsity": best_per_layer_sparsity,
        "compression_stats": comp_stats,
        "peak_gpu_mb": peak_gpu_mb,
    }


###############################################################################
#                           Argument Parsing                                  #
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="BERT Ternary Quantization on GLUE")

    # Task & model
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["cola", "mrpc", "rte", "stsb", "sst2", "qnli"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["fp", "ttq", "pttq", "ema_pttq"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")

    # Training
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="LR for non-quantized model params and FP weight copies")
    parser.add_argument("--lr_quant", type=float, default=5e-6,
                        help="LR for scaling factors (and thresholds if --lr_thresh not set)")
    parser.add_argument("--lr_thresh", type=float, default=None,
                        help="LR for thresholds only (overrides lr_quant for thresholds)")
    parser.add_argument("--optimizer_thresh", type=str, default="adamax",
                        choices=["adamax", "sgd"],
                        help="Optimizer for thresholds (adamax or sgd)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Quantized training epochs (or FP epochs if method=fp)")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="FP warmup epochs before applying quantization. "
                             "Total epochs = warmup_epochs + epochs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--scheduler_type", type=str, default="linear",
                        choices=["linear", "constant_with_warmup", "cosine"],
                        help="LR scheduler for non-quantized params")

    # Quantization
    parser.add_argument("--alpha", type=float, default=10000.0,
                        help="Sigmoid steepness (CRITICAL: keep at 10000)")
    parser.add_argument("--t", type=float, default=0.05,
                        help="TTQ fixed threshold factor")
    parser.add_argument("--init_x", type=float, default=1.0,
                        help="Initial threshold param a (pTTQ/EMA-pTTQ)")
    parser.add_argument("--init_y", type=float, default=1.0,
                        help="Initial threshold param b (pTTQ/EMA-pTTQ)")
    parser.add_argument("--k", type=float, default=1.0,
                        help="EMA tempering factor")
    parser.add_argument("--beta", type=float, default=0.9,
                        help="EMA decay factor")
    parser.add_argument("--smart_initial_scales", action="store_true",
                        help="Compute initial w_p/w_n from FP weight stats "
                             "(recommended for BERT/LayerNorm architectures)")

    # Infrastructure
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45",
                        help="Comma-separated seeds for Mo5 evaluation")

    return parser.parse_args()


###############################################################################
#                           Main                                              #
###############################################################################
def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    global SEEDS
    SEEDS = [int(s) for s in args.seeds.split(",")]

    start_time = time.time()
    all_results: List[Dict] = []

    for idx, seed in enumerate(SEEDS):
        print("=" * 80)
        print(f"Run {idx+1}/{len(SEEDS)} | seed={seed} | "
              f"method={args.method} | task={args.task_name}")
        print("=" * 80)
        res = run_single_seed(args, seed)
        all_results.append(res)

    total_time = time.time() - start_time

    # --- Aggregate: Median of N ---
    metric_keys = ["accuracy", "f1", "matthews_correlation", "pearson", "spearmanr"]
    median_metrics = {}
    for mk in metric_keys:
        vals = [r["best_metric_dict"].get(mk, float("nan")) for r in all_results]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        median_metrics[mk] = statistics.median(vals) if vals else float("nan")

    sparsities = [r["sparsity"] for r in all_results]
    median_sparsity = statistics.median(sparsities)

    # Aggregate per-layer sparsity (median across seeds)
    all_per_layer = [r.get("per_layer_sparsity", {}) for r in all_results]
    median_per_layer = {}
    if all_per_layer and all_per_layer[0]:
        for layer_name in all_per_layer[0]:
            vals = [pl.get(layer_name, 0.0) for pl in all_per_layer]
            median_per_layer[layer_name] = statistics.median(vals)

    # Aggregate compression stats
    comp_stats = all_results[0].get("compression_stats", {}) if all_results else {}
    peak_gpu_mb = max(r.get("peak_gpu_mb", 0) for r in all_results)

    # --- Print Summary ---
    print("\n" + "=" * 80)
    print(f"RESULTS: {args.method.upper()} on {args.task_name.upper()} "
          f"(median of {len(SEEDS)} seeds)")
    prim_key = _METRIC_FOR_TASK[args.task_name]
    print(f"  {prim_key}: {median_metrics.get(prim_key, float('nan')):.4f}")
    for mk, mv in median_metrics.items():
        if mk != prim_key and not math.isnan(mv):
            print(f"  {mk}: {mv:.4f}")
    print(f"  global_sparsity: {median_sparsity:.1%}")
    if median_per_layer:
        sp_vals = list(median_per_layer.values())
        print(f"  per_layer_sparsity: min={min(sp_vals):.1%} max={max(sp_vals):.1%} "
              f"mean={statistics.mean(sp_vals):.1%}")
    if comp_stats:
        print(f"  compression_ratio: {comp_stats.get('compression_ratio', 0):.1f}x "
              f"({comp_stats.get('quant_params', 0):,} quantized params)")
    print(f"  peak_gpu_mb: {peak_gpu_mb:.0f}")
    print(f"  total time: {total_time:.0f}s")
    # Per-seed breakdown
    print(f"\n  Per-seed {prim_key}:")
    for i, (seed, res) in enumerate(zip(SEEDS, all_results)):
        v = res["best_metric_dict"].get(prim_key, float("nan"))
        print(f"    seed {seed}: {v:.4f} (sparsity={res['sparsity']:.1%})")
    print("=" * 80)

    # --- Save to CSV (thread-safe with file locking) ---
    result_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_name": args.task_name,
        "method": args.method,
        "model_name": args.model_name,
        "lr": args.lr,
        "lr_quant": args.lr_quant,
        "lr_thresh": args.lr_thresh if args.lr_thresh is not None else args.lr_quant,
        "optimizer_thresh": args.optimizer_thresh,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs if args.method != "fp" else "N/A",
        "epochs": args.epochs,
        "alpha": args.alpha if args.method != "fp" else "N/A",
        "t": args.t if args.method == "ttq" else "N/A",
        "init_x": args.init_x if args.method in ("pttq", "ema_pttq") else "N/A",
        "init_y": args.init_y if args.method in ("pttq", "ema_pttq") else "N/A",
        "k": args.k if args.method in ("pttq", "ema_pttq") else "N/A",
        "beta": args.beta if args.method == "ema_pttq" else "N/A",
        "smart_initial_scales": args.smart_initial_scales,
        "accuracy": median_metrics.get("accuracy", ""),
        "f1": median_metrics.get("f1", ""),
        "matthews_correlation": median_metrics.get("matthews_correlation", ""),
        "pearson": median_metrics.get("pearson", ""),
        "spearmanr": median_metrics.get("spearmanr", ""),
        "global_sparsity": f"{median_sparsity:.4f}",
        "per_layer_sparsity": json.dumps(
            {k: f"{v:.4f}" for k, v in median_per_layer.items()}) if median_per_layer else "",
        "compression_ratio": f"{comp_stats.get('compression_ratio', 0):.2f}" if comp_stats else "",
        "quant_params": comp_stats.get("quant_params", ""),
        "total_params": comp_stats.get("total_params", ""),
        "peak_gpu_mb": f"{peak_gpu_mb:.0f}",
        "total_time_sec": f"{total_time:.0f}",
        "seeds": ",".join(map(str, SEEDS)),
    }

    columns = list(result_row.keys())
    # Thread-safe CSV append using file locking
    with open(RESULTS_FILE, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            file_exists = os.path.getsize(RESULTS_FILE) > 0
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_row)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    logger.info(f"Results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
