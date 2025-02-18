import argparse
import pickle
import glob
import numpy as np

def process_file(file_path):
    with open(file_path, 'rb') as pf:
        params = pickle.load(pf)
    
    test_mcc = params['TestMccPerEpoch']
    sparsity_rate = params['SparsityRatePerEpoch']
    
    best_metric = max(test_mcc)
    best_epoch = test_mcc.index(best_metric)
    best_sparsity = sparsity_rate[best_epoch]
    
    return best_metric, best_epoch + 1, best_sparsity  # +1 for 1-indexed epochs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parameters_pth_file', required=True, help="Parameters for the experiment. It must be a pth file (not json)", type=str)
    args = vars(ap.parse_args())

    parameters_file_pattern = args['parameters_pth_file']
    matching_files = glob.glob(parameters_file_pattern)

    if not matching_files:
        print(f"No files found matching the pattern: {parameters_file_pattern}")
        return

    all_best_metrics = []
    all_best_epochs = []
    all_best_sparsities = []

    print("Results per repetition:")
    print("=======================")

    for parameters_file in matching_files:
        best_metric, best_epoch, best_sparsity = process_file(parameters_file)
        all_best_metrics.append(best_metric)
        all_best_epochs.append(best_epoch)
        all_best_sparsities.append(best_sparsity)

        print(f"File: {parameters_file}")
        print(f"Best metric: {best_metric:.4f}")
        print(f"Epoch of best metric: {best_epoch}")
        print(f"Sparsity at best metric: {best_sparsity:.6f}")
        print()

    avg_best_metric = np.mean(all_best_metrics) * 100
    std_best_metric = np.std(all_best_metrics)  * 100
    avg_convergence_epoch = np.mean(all_best_epochs)
    avg_best_sparsity = np.mean(all_best_sparsities) * 100
    std_best_sparsity = np.std(all_best_sparsities) * 100

    print("Summary across all repetitions:")
    print("================================")
    print(f"Average best metric: {avg_best_metric:.2f} ± {std_best_metric:.2f}")
    print(f"Average time to convergence: {avg_convergence_epoch:.2f} epochs")
    print(f"Average sparsity at best metric: {avg_best_sparsity:.2f} ± {std_best_sparsity:.2f}")

if __name__ == "__main__":
    main()
