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
    original_keys = []  # New list to store original keys
    
    del json_data["1.0"]
    del json_data["TTQ"]
    del json_data["pTTQ"]

    for k, values in json_data.items():
        avg_sparsity = values["average_sparsity_at_best_metric"]["avg"]
        avg_mcc = values["average_best_metric"]["avg"]
        std_mcc = values["average_best_metric"]["std"]

        sparsity.append(avg_sparsity)
        mcc.append(avg_mcc)
        mcc_std.append(std_mcc)
        original_keys.append(k)  # Store the original key

        # Assign labels based on key
        if k in ["TTQ", "pTTQ"]:
            labels.append(k)
        else:
            labels.append("EMA PTTQ")  # Force label for anything else

    return np.array(sparsity), np.array(mcc), np.array(mcc_std), labels, original_keys

def plot_pareto_frontier(sparsity, mcc, mcc_std, output_file, labels, original_keys):
    """Plot the Pareto frontier of sparsity vs MCC with error bars."""

    # Sort by sparsity
    sorted_indices = np.argsort(sparsity)
    sparsity_sorted = sparsity[sorted_indices]
    mcc_sorted = mcc[sorted_indices]
    mcc_std_sorted = mcc_std[sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]
    original_keys_sorted = [original_keys[i] for i in sorted_indices]

    # Pareto frontier (sorting by sparsity but finding decreasing MCC values)
    pareto_indices = np.argsort(-mcc_sorted)
    pareto_frontier = mcc_sorted[pareto_indices]
    pareto_sparsity = sparsity_sorted[pareto_indices]
    pareto_std = mcc_std_sorted[pareto_indices]
    pareto_labels = [labels_sorted[i] for i in pareto_indices]
    pareto_original_keys = [original_keys_sorted[i] for i in pareto_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a consistent color mapping for labels
    label_colors = {
        "TTQ": "blue",
        "pTTQ": "orange",
        "EMA PTTQ": "green"
    }

    # Plotting with error bars
    for label in set(pareto_labels):
        indices = [i for i, l in enumerate(pareto_labels) if l == label]
        points = ax.errorbar(
            pareto_sparsity[indices],
            pareto_frontier[indices],
            yerr=pareto_std[indices],
            fmt='o',
            label=label,
            color=label_colors.get(label, "black"),  # Default to black if not in dict
            capsize=5
        )
        
        # Add annotations for EMA PTTQ points
        if label == "EMA PTTQ":
            for i in indices:
                ax.annotate(pareto_original_keys[i], 
                            (pareto_sparsity[i], pareto_frontier[i]),
                            xytext=(5, 5), textcoords='offset points')

    ax.set_title('Sparsity vs MCC for EMA-pTTQ at various values of $\\beta$')
    ax.set_xlabel('Average Sparsity (%)')
    ax.set_ylabel('Average MCC (%)')
    ax.grid()

    # Adjust the plot layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Place the legend outside the plot at the bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Save the plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot Pareto frontier of sparsity vs MCC from a JSON file.')
    parser.add_argument('input_file', type=str, help='Input JSON file containing results')

    args = parser.parse_args()

    # Load JSON data
    json_data = load_json(args.input_file)

    # Extract sparsity, MCC values, and original keys
    sparsity, mcc, mcc_std, labels, original_keys = extract_data(json_data)

    # Prepare output file name
    output_file = f"./results/beta_ablation_{args.input_file.split('/')[-1].replace('.json', '')}.png"

    # Plot Pareto frontier
    plot_pareto_frontier(sparsity, mcc, mcc_std, output_file, labels, original_keys)

if __name__ == '__main__':
    main()

