from tabulate import tabulate
import os
import pickle
import numpy as np
from src.utils.nbBitsStoreModel import get_nb_bits_model

def get_compression_gains(exp_folder_model_a, is_model_a_ternarized, exp_folder_model_b, is_model_b_ternarized):
    list_nb_bits_total_model_a, list_nb_bits_quantized_layers_model_a = get_nb_bits_model(exp_folder_model_a, is_model_a_ternarized)
    list_nb_bits_total_model_b, list_nb_bits_quantized_layers_model_b = get_nb_bits_model(exp_folder_model_b, is_model_b_ternarized)

    compression_gains_whole = []
    compression_gains_quantized = []
    for i in range(len(list_nb_bits_total_model_a)):
        nb_bits_whole_a = list_nb_bits_total_model_a[i]
        nb_bits_whole_b = list_nb_bits_total_model_b[i]
        nb_bits_quantized_a = list_nb_bits_quantized_layers_model_a[i]
        nb_bits_quantized_b = list_nb_bits_quantized_layers_model_b[i]

        compression_gains_whole.append(1 - (nb_bits_whole_a / nb_bits_whole_b))
        compression_gains_quantized.append(1 - (nb_bits_quantized_a / nb_bits_quantized_b))

    return (np.mean(compression_gains_whole), np.std(compression_gains_whole)),(np.mean(compression_gains_quantized), np.std(compression_gains_quantized))

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

            (global_gain, global_std), (local_gain, local_std) = get_compression_gains(exp_folder_model_a, is_model_a_ternarized, exp_folder_model_b, is_model_b_ternarized)

            global_row.append(f"{global_gain:.2%} ± {global_std:.2%}")
            local_row.append(f"{local_gain:.2%} ± {local_std:.2%}")

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
