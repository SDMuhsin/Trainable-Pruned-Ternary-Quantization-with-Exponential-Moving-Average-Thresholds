'''
    How to use 
     python3 src/utils/getKOverrideResults.py --base_pattern=./results/CameraReady_SVHN_RESNET18_experimental_k{}_OW_0/metrics/results_exp-CameraReady_SVHN_RESNET18_experimental_k{}_rep-*.pth
     The {} tells the code to iterate through all values of k, the * tells the code to iterate through all repetitions

'''


import argparse
import pickle
import glob
import numpy as np
import json
import os
import re

def process_file(file_path):
    with open(file_path, 'rb') as pf:
        params = pickle.load(pf)

    test_mcc = params['TestMccPerEpoch']
    sparsity_rate = params['SparsityRatePerEpoch']

    best_metric = max(test_mcc)
    best_epoch = test_mcc.index(best_metric)
    best_sparsity = sparsity_rate[best_epoch]

    return best_metric, best_epoch + 1, best_sparsity

def process_k_value(file_pattern):
    matching_files = glob.glob(file_pattern)

    if not matching_files:
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
        "average_best_metric": {"avg": avg_best_metric, "std": std_best_metric},
        "average_time_to_convergence": {"avg": avg_convergence_epoch},
        "average_sparsity_at_best_metric": {"avg": avg_best_sparsity, "std": std_best_sparsity}
    }

def consolidate_results(base_pattern):
    consolidated_results = {}
    
    # Extract prefix for naming the JSON file
    match = re.search(r'CameraReady_(.+?)_experimental_k1.0_beta', base_pattern)
    prefix = match.group(1) if match else "results"

    for k in np.arange(0.1, 1.1, 0.1):
        k_rounded = round(k, 1)
        file_pattern = base_pattern.format(k_rounded, k_rounded)
        results = process_k_value(file_pattern)
        if results:
            consolidated_results[str(k_rounded)] = results
        else:
            print(f"No files found for k={k_rounded}.")

    return consolidated_results, prefix
'''
    How to use 
     python3 src/utils/getKOverrideResults.py --base_pattern=./results/CameraReady_SVHN_RESNET18_experimental_k{}_OW_0/metrics/results_exp-CameraReady_SVHN_RESNET18_experimental_k{}_rep-*.pth
     The {} tells the code to iterate through all values of k, the * tells the code to iterate through all repetitions

'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_pattern', required=True, help="Base pattern for file search.")
    
    args = parser.parse_args()
    
    os.makedirs('./results', exist_ok=True)
    
    consolidated_results, prefix = consolidate_results(args.base_pattern)
    
    results_json_path = f'./results/beta_ablation_results_{prefix}.json'
    
    with open(results_json_path, 'w') as json_file:
        json.dump(consolidated_results, json_file, indent=4)
    
    print(f'Results saved to {results_json_path}')

if __name__ == "__main__":
    main()
