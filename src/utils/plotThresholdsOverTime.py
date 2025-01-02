import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

def safely_convert_to_float(x):
    """
    Safely convert x to a Python float, regardless of whether x is:
      - a torch.Tensor
      - a NumPy array
      - a Python float
      - or anything else that can be cast to float.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        else:
            return float(x[0])
    return float(x)

def convert_list_to_floats(any_list):
    """Convert each element of a list to Python float."""
    return [safely_convert_to_float(item) for item in any_list]

def smoothness_metric(values, gap_fraction=0.01):
    """
    Calculate a measure of smoothness based on pairs of values that are
    'gap_fraction' of the time-steps apart.
    
    If there are M points and gap = int(gap_fraction * M),
    we compute:
        diffs = [|y_{i + gap} - y_i| for i in range(M - gap)]
    and then average those diffs.
    
    S = (1 / (M - gap)) * sum_{i=0 to M-gap-1} |y_{i + gap} - y_i|
    """
    M = len(values)
    if M < 2:
        return 0.0
    
    gap = int(gap_fraction * M)
    # Ensure gap is at least 1
    if gap < 1:
        gap = 1
    
    if M <= gap:
        # If the list is too short to do i+gap, return 0
        return 0.0
    
    diffs = [abs(values[i + gap] - values[i]) for i in range(M - gap)]
    return sum(diffs) / len(diffs)

def main():
    parser = argparse.ArgumentParser(description="Plot the evolution of delta_mins and delta_maxes.")
    parser.add_argument("--tensor_index", type=int, default=0,
                        help="Index of the tensor to plot (0 or 1).")
    parser.add_argument("--smooth_gap", type=float, default=0.01,
                        help="Fraction of steps for the smoothness measure (default=0.01).")
    args = parser.parse_args()
    
    tensor_key = str(args.tensor_index)  # "0" or "1"

    # --- Load pTTQ data ---
    data_pttq = np.load("./results/track_thresholds_pttq.npy", allow_pickle=True).item()
    pttq_delta_mins_raw = data_pttq['delta_mins'][tensor_key]
    pttq_delta_maxes_raw = data_pttq['delta_maxes'][tensor_key]
    
    pttq_delta_mins = convert_list_to_floats(pttq_delta_mins_raw)
    pttq_delta_maxes = convert_list_to_floats(pttq_delta_maxes_raw)

    # --- Load EMA-pTTQ data ---
    data_exp = np.load("./results/track_thresholds_experimental.npy", allow_pickle=True).item()
    exp_delta_mins_raw = data_exp['delta_mins'][tensor_key]
    exp_delta_maxes_raw = data_exp['delta_maxes'][tensor_key]
    
    exp_delta_mins = convert_list_to_floats(exp_delta_mins_raw)
    exp_delta_maxes = convert_list_to_floats(exp_delta_maxes_raw)

    # --- Remove first 1% of the data ---
    # pTTQ
    n_pttq = len(pttq_delta_mins)
    start_pttq = int(0.01 * n_pttq)
    pttq_delta_mins = pttq_delta_mins[start_pttq:]
    pttq_delta_maxes = pttq_delta_maxes[start_pttq:]

    # EMA-pTTQ
    n_exp = len(exp_delta_mins)
    start_exp = int(0.01 * n_exp)
    exp_delta_mins = exp_delta_mins[start_exp:]
    exp_delta_maxes = exp_delta_maxes[start_exp:]

    # --- Build new x-axes [0, 100] ---
    num_steps_pttq = len(pttq_delta_mins)
    x_pttq = np.linspace(0, 100, num_steps_pttq) if num_steps_pttq > 0 else []

    num_steps_exp = len(exp_delta_mins)
    x_exp = np.linspace(0, 100, num_steps_exp) if num_steps_exp > 0 else []

    # --- Compute smoothness metrics with gap_fraction ---
    gap_frac = args.smooth_gap
    smooth_pttq_min = smoothness_metric(pttq_delta_mins, gap_fraction=gap_frac)
    smooth_pttq_max = smoothness_metric(pttq_delta_maxes, gap_fraction=gap_frac)
    smooth_exp_min = smoothness_metric(exp_delta_mins, gap_fraction=gap_frac)
    smooth_exp_max = smoothness_metric(exp_delta_maxes, gap_fraction=gap_frac)

    print(f"Smoothness values (gap_fraction={gap_frac:.2%}):")
    print(f"  pTTQ Δ_min : {smooth_pttq_min:.6f}")
    print(f"  pTTQ Δ_max : {smooth_pttq_max:.6f}")
    print(f"  EMA-pTTQ Δ_min : {smooth_exp_min:.6f}")
    print(f"  EMA-pTTQ Δ_max : {smooth_exp_max:.6f}")

    # ------------------------------------------------------------------
    # NEW BLOCK: Normalize each curve to start at 0 (AFTER smoothness)
    # ------------------------------------------------------------------
    if num_steps_pttq > 0:
        offset_pttq_min = pttq_delta_mins[0]
        offset_pttq_max = pttq_delta_maxes[0]
        pttq_delta_mins = [v - offset_pttq_min for v in pttq_delta_mins]
        pttq_delta_maxes = [v - offset_pttq_max for v in pttq_delta_maxes]

    if num_steps_exp > 0:
        offset_exp_min = exp_delta_mins[0]
        offset_exp_max = exp_delta_maxes[0]
        exp_delta_mins = [v - offset_exp_min for v in exp_delta_mins]
        exp_delta_maxes = [v - offset_exp_max for v in exp_delta_maxes]
    # ------------------------------------------------------------------

    # --- Plot ---
    plt.figure(figsize=(7, 5))

    # pTTQ
    if num_steps_pttq > 0:
        plt.plot(x_pttq, pttq_delta_mins, label='pTTQ $\\Delta_{\\min}$', color='blue')
        plt.plot(x_pttq, pttq_delta_maxes, label='pTTQ $\\Delta_{\\max}$', color='blue', linestyle='--')

    # EMA-pTTQ
    if num_steps_exp > 0:
        plt.plot(x_exp, exp_delta_mins, label='EMA-pTTQ $\\Delta_{\\min}$', color='orange')
        plt.plot(x_exp, exp_delta_maxes, label='EMA-pTTQ $\\Delta_{\\max}$', color='orange', linestyle='--')

    plt.xlabel("Pruning step (%)")
    plt.ylabel("Threshold value")
    plt.title(f"Threshold evolution for Tensor {args.tensor_index}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./results/threshold_evolution.png")
    print("Saved plot to ./results/threshold_evolution.png")


if __name__ == "__main__":
    main()

