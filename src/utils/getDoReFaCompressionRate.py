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
from src.utils.nbBitsStoreModel import get_doReFa_nb_bits_storage
from src.utils.nbBitsStoreModel import compute_doReFa_compression_gains

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--exp_folder', required=True, help="Path to the experiment folder of the first model", type=str)
    args = vars(ap.parse_args())

    exp_results_folder  = args["exp_folder"]

    # Step 1: Calculate storage info
    storage_info = get_doReFa_nb_bits_storage(exp_results_folder)

    # Step 2: Compute compression gains
    compression_gains = compute_doReFa_compression_gains(storage_info)

    print(f"Global Compression Gain: {compression_gains['global_compression_gain']:.2f}%")
    print(f"Local Compression Gain: {compression_gains['local_compression_gain']:.2f}%")
    print(f"Quantized Percentage: {storage_info['quantized_percentage']:.2f}%")

if __name__=="__main__":
    main()
