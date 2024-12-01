import argparse
import pickle
import glob
import numpy as np
import json
import os

def process_file(file_path):
    with open(file_path, 'rb') as pf:
        params = pickle.load(pf)

    test_mcc = params['TestMccPerEpoch']
    sparsity_rate = params['SparsityRatePerEpoch']

    best_metric = max(test_mcc)
    best_epoch = test_mcc.index(best_metric)
    best_sparsity = sparsity_rate[best_epoch]

    return best_metric, best_epoch + 1, best_sparsity  # +1 for 1-indexed epochs

def process_k_value(file_pattern):
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No files found matching the pattern: {file_pattern}")
        return None

    all_best_metrics = []
    all_best_epochs = []
    all_best_sparsities = []

    for parameters_file in matching_files:
        best_metric, best_epoch, best_sparsity = process_file(parameters_file)
        all_best_metrics.append(best_metric)
        all_best_epochs.append(best_epoch)
        all_best_sparsities.append(best_sparsity)

    avg_best_metric = np.mean(all_best_metrics) * 100
    std_best_metric = np.std(all_best_metrics) * 100
    avg_convergence_epoch = np.mean(all_best_epochs)
    avg_best_sparsity = np.mean(all_best_sparsities) * 100
    std_best_sparsity = np.std(all_best_sparsities) * 100

    return {
        "avg_best_metric": f"{avg_best_metric:.2f} ± {std_best_metric:.2f}",
        "avg_convergence_epoch": f"{avg_convergence_epoch:.2f}",
        "avg_best_sparsity": f"{avg_best_sparsity:.2f} ± {std_best_sparsity:.2f}"
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parameters_pth_file', required=True, help="Parameters for the experiment. Use # for k value placeholder.", type=str)
    args = vars(ap.parse_args())

    base_pattern = args['parameters_pth_file']
    k_values = [round(k, 1) for k in np.arange(0.1, 2.1, 0.1)]
    
    results = {}

    for k in k_values:
        file_pattern = base_pattern.replace('#', str(k))
        k_results = process_k_value(file_pattern)
        if k_results:
            results[str(k)] = k_results

    # Extract experiment name from the file pattern
    exp_name = os.path.basename(base_pattern).split('_exp-')[1].split('_k#')[0]
    output_file = f"./results/koverride_results_{exp_name}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
