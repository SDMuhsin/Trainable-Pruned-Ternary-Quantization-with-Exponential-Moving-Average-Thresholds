"""
SparseGPT: One-shot pruning (+ optional quantization) using approximate
second-order information.

Adapted from https://github.com/IST-DASLab/sparsegpt (Frantar & Alistarh, 2023).
Original paper: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"

The Quantizer, quantize(), SparseGPT.add_batch(), and SparseGPT.fasterprune()
are kept faithful to the original repo.  Only model-specific scaffolding
(apply_sparsegpt_to_bert) is new.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


###############################################################################
#  Quantizer — verbatim from original repo quant.py                          #
#  (only whitespace / type-hint changes)                                     #
###############################################################################

def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self, bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        grouprows=1
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.grouprows = grouprows

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.grouprows > 1:
                    x = x.reshape((x.shape[0] // self.grouprows, -1))
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            if self.grouprows > 1:
                self.scale = self.scale.unsqueeze(1).repeat(1, self.grouprows)
                self.zero = self.zero.unsqueeze(1).repeat(1, self.grouprows)
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


###############################################################################
#  SparseGPT — faithful to original repo sparsegpt.py                       #
#  Differences vs upstream:                                                   #
#    - Removed Conv2d / transformers.Conv1D paths (BERT is Linear-only)      #
#    - Removed N:M structured pruning (prunen/prunem) — unstructured only    #
#    - Removed DEBUG scaffolding                                              #
###############################################################################

class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


###############################################################################
#  BERT-specific scaffolding (new code)                                      #
###############################################################################

def find_linear_layers(module: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
    """Recursively find all nn.Linear layers in a module."""
    result = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            result[full_name] = child
        else:
            result.update(find_linear_layers(child, full_name))
    return result


@torch.no_grad()
def apply_sparsegpt_to_bert(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sparsity: float = 0.5,
    nsamples: int = 128,
    blocksize: int = 128,
    percdamp: float = 0.01,
    wbits: int = 16,
) -> Dict[str, float]:
    """Apply SparseGPT one-shot pruning (+ optional quantization) to BERT.

    Uses a hook-based approach: runs calibration data through the full model
    to collect Hessian information for all encoder Linear layers, then prunes
    (and optionally quantizes) each layer independently.

    Args:
        model: HuggingFace BertForSequenceClassification (or similar)
        dataloader: calibration data (training set DataLoader)
        device: torch device
        sparsity: target sparsity (fraction of zeros)
        nsamples: number of calibration samples to use
        blocksize: SparseGPT block size
        percdamp: Hessian dampening percentage
        wbits: weight bit-width for quantization (16 = no quantization)

    Returns:
        Dict with per-layer actual sparsity.
    """
    model.eval()

    # Find all Linear layers in the encoder (skip embeddings and classifier)
    layers_to_prune = {}
    for name, module in model.named_modules():
        if (isinstance(module, nn.Linear)
                and "encoder.layer" in name
                and "LayerNorm" not in name):
            layers_to_prune[name] = module

    quant_tag = f" + {wbits}-bit quantization" if wbits < 16 else ""
    print(f"[SparseGPT] Found {len(layers_to_prune)} Linear layers to prune"
          f"{quant_tag}")

    # Create SparseGPT objects for each layer (+ optional quantizer)
    gpts = {}
    for name, linear_mod in layers_to_prune.items():
        gpts[name] = SparseGPT(linear_mod)
        if wbits < 16:
            gpts[name].quantizer = Quantizer()
            gpts[name].quantizer.configure(
                wbits, perchannel=True, sym=False, mse=False)

    # Register forward hooks to collect Hessian info during calibration
    handles = []
    for name in gpts:
        def make_hook(n):
            def hook_fn(module, inp, out):
                gpts[n].add_batch(inp[0].data, out.data)
            return hook_fn
        handles.append(layers_to_prune[name].register_forward_hook(make_hook(name)))

    # Run calibration data through the full model to build all Hessians
    print(f"[SparseGPT] Collecting Hessian info ({nsamples} samples)...")
    sample_count = 0
    for batch in dataloader:
        if sample_count >= nsamples:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch)
        sample_count += batch["input_ids"].shape[0]

    # Remove hooks
    for h in handles:
        h.remove()
    print(f"[SparseGPT] Collected Hessian from {sample_count} samples")

    # Prune (+ quantize) each layer
    layer_stats = {}
    for name in sorted(gpts.keys()):
        rows, cols = gpts[name].rows, gpts[name].columns
        print(f"  Pruning {name} ({rows}x{cols})...")
        gpts[name].fasterprune(sparsity, blocksize=blocksize, percdamp=percdamp)

        # Record actual sparsity
        w = layers_to_prune[name].weight.data
        actual_sp = (w == 0).float().mean().item()
        layer_stats[name] = actual_sp
        gpts[name].free()

    print(f"\n[SparseGPT] Pruning complete. {len(layer_stats)} layers pruned.")
    return layer_stats
