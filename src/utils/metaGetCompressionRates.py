import os
import subprocess
import numpy as np
from tabulate import tabulate

# Define the datasets and techniques
datasets = ["MNIST_2D_CNN", "FMNIST_RESNET18", "KMNIST_RESNET18", "EMNIST_RESNET18",
            "SVHN_RESNET18", "CIFAR10_RESNET50", "CIFAR100_RESNET50", "STL10_RESNET50"]
techniques = ["TTQ", "PTTQ", "experimental_k1"]

# Define the base paths for model A and B
base_path_model_a = "./results/CameraReady_{}_FP_OW_0/"
base_path_model_b = "./results/CameraReady_{}_{}_OW_0/"

# Initialize tables for global and local compression rates
global_compression_results = []
local_compression_results = []

# Loop through datasets and techniques
for dataset in datasets:
    global_row = [dataset]
    local_row = [dataset]
    for technique in techniques:
        # Adjust technique folder name if necessary
        technique_folder = technique if technique != "experimental_k1" else "experimental_k1.0"

        # Construct folder paths
        exp_folder_model_a = base_path_model_a.format(dataset)
        exp_folder_model_b = base_path_model_b.format(dataset, technique_folder)

        # Construct the command
        command = [
            "python3", "src/utils/getCompressionRate.py",
            "--exp_folder_model_a", exp_folder_model_a,
            "--is_model_a_ternarized", "False",
            "--exp_folder_model_b", exp_folder_model_b,
            "--is_model_b_ternarized", "True"
        ]

        try:
            # Run the command and capture output
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
            
            # Parse compression rates from the script's output
            global_rate = None
            local_rate = None
            for line in output.splitlines():
                if "Compression rate (WHOLE)" in line:
                    global_rate = float(line.split(":")[1].split("+-")[0].strip())
                elif "Compression rate (QUANTIZED LAYERS ONLY)" in line:
                    local_rate = float(line.split(":")[1].split("+-")[0].strip())

            # Append results
            global_row.append(global_rate if global_rate is not None else "N/A")
            local_row.append(local_rate if local_rate is not None else "N/A")
        
        except subprocess.CalledProcessError as e:
            print(f"Error while processing {dataset} with {technique}: {e.output}")
            global_row.append("Error")
            local_row.append("Error")

    # Add rows to the tables
    global_compression_results.append(global_row)
    local_compression_results.append(local_row)

# Define headers
headers = ["Dataset"] + techniques

# Print tables
print("\nGlobal Compression Rates:")
print(tabulate(global_compression_results, headers=headers, tablefmt="grid"))

print("\nLocal Compression Rates:")
print(tabulate(local_compression_results, headers=headers, tablefmt="grid"))

