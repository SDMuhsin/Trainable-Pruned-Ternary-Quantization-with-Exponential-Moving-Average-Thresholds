from tabulate import tabulate
import os
import pickle
import numpy as np
from src.utils.nbBitsStoreModel import get_nb_bits_model

def get_compression_rates(exp_folder_model_a, is_model_a_ternarized, exp_folder_model_b, is_model_b_ternarized):
    list_nb_bits_total_model_a, list_nb_bits_quantized_layers_model_a = get_nb_bits_model(exp_folder_model_a, is_model_a_ternarized)
    list_nb_bits_total_model_b, list_nb_bits_quantized_layers_model_b = get_nb_bits_model(exp_folder_model_b, is_model_b_ternarized)

    compression_rates_whole = []
    compression_rates_quantized = []
    for i in range(len(list_nb_bits_total_model_a)):
        nb_bits_whole_a = list_nb_bits_total_model_a[i]
        nb_bits_whole_b = list_nb_bits_total_model_b[i]
        nb_bits_quantized_a = list_nb_bits_quantized_layers_model_a[i]
        nb_bits_quantized_b = list_nb_bits_quantized_layers_model_b[i]

        compression_rates_whole.append(nb_bits_whole_b / nb_bits_whole_a)
        compression_rates_quantized.append(nb_bits_quantized_b / nb_bits_quantized_a)

    return compression_rates_whole, compression_rates_quantized

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

            global_rates, local_rates = get_compression_rates(exp_folder_model_a, is_model_a_ternarized, exp_folder_model_b, is_model_b_ternarized)

            # Convert rates to gains
            global_gains = [1 - rate for rate in global_rates]
            local_gains = [1 - rate for rate in local_rates]
            
            global_gain_mean = np.mean(global_gains)
            global_gain_std = np.std(global_gains)
            local_gain_mean = np.mean(local_gains)
            local_gain_std = np.std(local_gains)

            global_row.append(f"{global_gain_mean:.2%} ± {global_gain_std:.2%}")
            local_row.append(f"{local_gain_mean:.2%} ± {local_gain_std:.2%}")

        global_results.append(global_row)
        local_results.append(local_row)

    global_table = tabulate(global_results, headers=datasets, showindex=techniques, tablefmt="grid")
    local_table = tabulate(local_results, headers=datasets, showindex=techniques, tablefmt="grid")

    print("Global Compression Gains:")
    print(global_table)
    print("\nLocal Compression Gains:")
    print(local_table)

if __name__ == "__main__":
    main()
