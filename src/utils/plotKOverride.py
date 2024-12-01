import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_json(file_name):
    """Load JSON data from a file."""
    with open(file_name, 'r') as f:
        return json.load(f)

def extract_data(json_data):
    """Extract sparsity, MCC, and standard deviation values from JSON data."""
    sparsity = []
    mcc = []
    mcc_std = []
    labels = []

    for k, values in json_data.items():
        avg_sparsity = values["average_sparsity_at_best_metric"]["avg"]
        avg_mcc = values["average_best_metric"]["avg"]
        std_mcc = values["average_best_metric"]["std"]

        sparsity.append(avg_sparsity)
        mcc.append(avg_mcc)
        mcc_std.append(std_mcc)

        # Assign labels based on key
        if k in ["TTQ", "pTTQ"]:
            labels.append(k)
        else:
            labels.append("EMA PTTQ")
    
    return np.array(sparsity), np.array(mcc), np.array(mcc_std), labels

def plot_pareto_frontier(sparsity, mcc, mcc_std, output_file, labels):
    """Plot the Pareto frontier of sparsity vs MCC with error bars."""
    # Sort by sparsity
    sorted_indices = np.argsort(sparsity)
    sparsity_sorted = sparsity[sorted_indices]
    mcc_sorted = mcc[sorted_indices]
    mcc_std_sorted = mcc_std[sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]

    # Find the Pareto frontier
    pareto_frontier = mcc_sorted[np.argsort(-mcc_sorted)]
    pareto_sparsity = sparsity_sorted[np.argsort(-mcc_sorted)]
    pareto_std = mcc_std_sorted[np.argsort(-mcc_sorted)]
    
    plt.figure(figsize=(10, 6))

    # Plotting with error bars
    for label in set(labels_sorted):
        indices = [i for i, l in enumerate(labels_sorted) if l == label]
        plt.errorbar(pareto_sparsity[indices], pareto_frontier[indices], 
                     yerr=pareto_std[indices], fmt='o', label=label, capsize=5)

    plt.title('Pareto Frontier of Sparsity vs MCC')
    plt.xlabel('Average Sparsity at Best Metric (%)')
    plt.ylabel('Average Best Metric (MCC)')
    plt.grid()
    plt.legend()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot Pareto frontier of sparsity vs MCC from a JSON file.')
    parser.add_argument('input_file', type=str, help='Input JSON file containing results')
    
    args = parser.parse_args()
    
    # Load JSON data
    json_data = load_json(args.input_file)
    
    # Extract sparsity and MCC values
    sparsity, mcc, mcc_std, labels = extract_data(json_data)
    
    # Prepare output file name
    output_file = f"./results/pareto_frontier_{args.input_file.split('/')[-1].replace('.json', '')}.png"
    
    # Plot Pareto frontier
    plot_pareto_frontier(sparsity, mcc, mcc_std, output_file, labels)
    
if __name__ == '__main__':
    main()
