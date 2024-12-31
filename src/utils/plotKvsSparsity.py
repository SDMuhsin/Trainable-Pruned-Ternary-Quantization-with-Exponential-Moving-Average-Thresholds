import json
import matplotlib.pyplot as plt
import sys

def plot_k_vs_sparsity(file_path):
    # Load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    del data['pTTQ']
    del data['TTQ']

    # Extract k values and corresponding sparsities
    k_values = []
    sparsities = []
    sparsity_errors = []
    
    for k, values in data.items():
        k_values.append(float(k))
        sparsity = values['average_sparsity_at_best_metric']['avg']
        sparsities.append(sparsity)
        sparsity_errors.append(values['average_sparsity_at_best_metric']['std'])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, sparsities, yerr=sparsity_errors, fmt='o-', capsize=5)
    
    # Set labels with increased font size
    plt.xlabel('k', fontsize=16)  # Quadrupled font size from default (usually ~4)
    plt.ylabel('Sparsity', fontsize=16)  # Quadrupled font size from default (usually ~4)
    
    plt.title('k vs Sparsity for MNIST with 2D CNN', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('./results/k_vs_sparsity_mnist.png')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_json_file>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    plot_k_vs_sparsity(json_file_path)

