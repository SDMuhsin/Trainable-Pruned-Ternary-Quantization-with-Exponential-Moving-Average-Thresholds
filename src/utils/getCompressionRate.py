#!/usr/bin/env python3
"""
    Computes the compression rate between two models A and B.
    The compression rate is defined as the ratio beteween the number of bits
    necessary to store the parameters of model A and the number of bits necessary
    to store the parameters of model B.

    Options:
    --------
    --exp_folder_model_a: str
        Path to the experiment folder of the first model.
    --exp_folder_model_b: str
        Path to the experiment folder of the second model.
"""
import os
import torch
import argparse
import pickle
import numpy as np
from src.utils.nbBitsStoreModel import get_nb_bits_model

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--exp_folder_model_a', required=True, help="Path to the experiment folder of the first model", type=str)
    ap.add_argument('--is_model_a_ternarized', required=True, help="True if the model a has been ternarized", type=str)
    ap.add_argument('--exp_folder_model_b', required=True, help="Path to the experiment folder of the second model", type=str)
    ap.add_argument('--is_model_b_ternarized', required=True, help="True if the model b has been ternarized", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    exp_folder_model_a = args['exp_folder_model_a']
    is_model_a_ternarized = args['is_model_a_ternarized']
    if (is_model_a_ternarized.lower() == "true"):
        is_model_a_ternarized = True
    else:
        is_model_a_ternarized = False
    exp_folder_model_b = args['exp_folder_model_b']
    is_model_b_ternarized = args['is_model_b_ternarized']
    if (is_model_b_ternarized.lower() == "true"):
        is_model_b_ternarized = True
    else:
        is_model_b_ternarized = False

    #==========================================================================#
    # Getting the number of bits necessary for the experiment
    list_nb_bits_total_model_a, list_nb_bits_quantized_layers_model_a = get_nb_bits_model(exp_folder_model_a, is_model_a_ternarized)
    list_nb_bits_total_model_b, list_nb_bits_quantized_layers_model_b = get_nb_bits_model(exp_folder_model_b, is_model_b_ternarized)

    # Getting the compression rates for the WHOLE model
    compression_rates_whole = []
    compression_rates_gains_whole = []
    for i in range(len(list_nb_bits_total_model_a)):
        nb_bits_whole_a = list_nb_bits_total_model_a[i]
        nb_bits_whole_b = list_nb_bits_total_model_b[i]
        compression_rates_whole.append(nb_bits_whole_b/nb_bits_whole_a)
        compression_rates_gains_whole.append(1-nb_bits_whole_b/nb_bits_whole_a)

    # Getting the compression rates for the QUANTIZED LAYERS only
    compression_rates_quantized = []
    compression_rates_gains_quantized = []
    for i in range(len(list_nb_bits_total_model_a)):
        nb_bits_quantized_a = list_nb_bits_quantized_layers_model_a[i]
        nb_bits_quantized_b = list_nb_bits_quantized_layers_model_b[i]
        # compression_rates_quantized.append(nb_bits_quantized_a/nb_bits_quantized_b)
        compression_rates_quantized.append(nb_bits_quantized_b/nb_bits_quantized_a)
        compression_rates_gains_quantized.append(1-nb_bits_quantized_b/nb_bits_quantized_a)


    # Getting the mean values
    print("\n\n\n\nCompression rate (WHOLE) between the two models: {} +- {} \n\n\n\n".format(np.mean(compression_rates_whole)*100, np.std(compression_rates_whole)*100))
    print("\tCompression rate GAIN (WHOLE) between the two models: {} +- {} \n\n\n\n".format(np.mean(compression_rates_gains_whole)*100, np.std(compression_rates_gains_whole)*100))
    print("\nCompression rate (QUANTIZED LAYERS ONLY) between the two models: {} +- {}\n\n\n\n".format(np.mean(compression_rates_quantized)*100, np.std(compression_rates_quantized)*100))
    print("\tCompression rate GAIN (QUANTIZED LAYERS ONLY) between the two models: {} +- {}\n\n\n\n".format(np.mean(compression_rates_gains_quantized)*100, np.std(compression_rates_gains_quantized)*100))
    #==========================================================================#
    # Storage of the results
    results_to_store = {
                            'CompressionRateTotal': compression_rates_whole,
                            'CompressionRateGainsTotal': compression_rates_gains_whole,
                            'CompressionRateQuantized': compression_rates_quantized,
                            'CompressionRateGainsQuantized': compression_rates_gains_quantized
                        }
    with open(exp_folder_model_b+'/compressionRates.pth', "wb") as fp:
        pickle.dump(results_to_store, fp)

if __name__=="__main__":
    main()
