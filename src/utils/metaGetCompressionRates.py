from tabulate import tabulate
import os
import pickle
import numpy as np

def get_compression_rates(exp_folder_model_a, is_model_a_ternarized, exp_folder_model_b, is_model_b_ternarized):
    # Simulate the get_nb_bits_model function
    def get_nb_bits_model(folder, is_ternarized):
        # Placeholder for actual implementation
        return np.random.randint(1000, 10000), np.random.randint(500, 5000)

    nb_bits_total_model_a, nb_bits_quantized_layers_model_a = get_nb_bits_model(exp_folder_model_a, is_model_a_ternarized)
    nb_bits_total_model_b, nb_bits_quantized_layers_model_b = get_nb_bits_model(exp_folder_model_b, is_model_b_ternarized)

    compression_rate_whole = nb_bits_total_b / nb_bits_total_a
    compression_rate_quantized = nb_bits_quantized_layers_b / nb_bits_quantized_layers_a

    return compression_rate_whole, compression_rate_quantized

def main():
    datasets = ["MNIST_2D_CNN", "FMNIST_RESNET18", "KMNIST_RESNET18", "EMNIST_RESNET18", "SVHN_RESNET18", "CIFAR10_RESNET50", "CIFAR100_RESNET50", "STL10_RESNET50"]
    techniques = ["TTQ", "PTTQ", "experimental_k1"]

    global_results = []
    local_results = []

    for technique in techniques:
        global_row = []
        local_row = []
        for dataset in datasets:
            exp_folder_model_a = f"./results/CameraReady_{dataset}_FP_OW_0/"
            exp_folder_model_b = f"./results/CameraReady_{dataset}_{technique}_OW_0/"
            
            if technique == "experimental_k1" and not os.path.exists(exp_folder_model_b):
                exp_folder_model_b = f"./results/CameraReady_{dataset}_experimental_k1.0_OW_0/"

            is_model_a_ternarized = False
            is_model_b_ternarized = True

            global_rate, local_rate = get_compression_rates(exp_folder_model_a, is_model_a_ternarized, exp_folder_model_b, is_model_b_ternarized)

            global_row.append(global_rate)
            local_row.append(local_rate)

        global_results.append(global_row)
        local_results.append(local_row)

    global_table = tabulate(global_results, headers=datasets, showindex=techniques, tablefmt="grid", floatfmt=".4f")
    local_table = tabulate(local_results, headers=datasets, showindex=techniques, tablefmt="grid", floatfmt=".4f")

    print("Global Compression Rates:")
    print(global_table)
    print("\nLocal Compression Rates:")
    print(local_table)

if __name__ == "__main__":
    main()
