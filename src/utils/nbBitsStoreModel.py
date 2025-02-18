#!/usr/bin/env python3
"""
    This code gives the number of bits necessary to store a model, with respect to two points
    of view:
    - The quantized layers ONLY.
    - The whole model.
    It takes into account the sparsity rate to do the computation

    Options:
    --------
    --exp_results_folder: str
        Path to the results folder of the experiment
    --is_model_ternarized: bool'
        True if the model has been ternarized. Default = True

"""
import os
import pickle
import argparse
import numpy as np
import torch
from src.utils.model_compression import get_params_groups_to_quantize


def nb_bits_store_tensor_csr(tensor, is_model_ternarized=False, ternarized=False):
    """
        Gives the number of bits necessary to store a tensor using the CSR
        sparse tensor formalism.
        PYTORCH DOC: The primary advantage of the CSR format over the COO format
        is better use of storage and much faster computation operations such as
        sparse matrix-vector multiplication using MKL and MAGMA backends.
        PROBLEM: IT HAS NOT BEEN IMPLEMENTED FOR MULTIDIMENSIONAL TENSORS !!!

        Arguments:
        ----------
        tensor: torch.tensor
            Tensor that we want to store
        is_model_ternarized: bool
            True if the model has been ternarized.
        ternarized: bool
            True if the tensor has been ternarized
    """
    # Encoding the tensort using CSR sparse storage
    raise NotImplementedError("\n\nCSR has not been implemented for multi-dimensional tensors\n\n")


def nb_bits_store_tensor_coo(tensor, is_model_ternarized=False, ternarized=False):
    """
        Gives the number of bits necessary to store a tensor using the COO
        sparse tensor formalism

        Arguments:
        ----------
        tensor: torch.tensor
            Tensor that we want to store
        is_model_ternarized: bool
            True if the model has been ternarized.
        ternarized: bool
            True if the tensor has been ternarized
    """
    # Encoding the tensort using COO sparse storage ONLY FOR THE QUANTIZED LAYERS
    nb_bits = -1
    if (ternarized) and (is_model_ternarized):
        # First, we flatten the tensor (VERY IMPORTANT TO REDUCE MEMORY CONSUMPTION)
        tensor_flatten = torch.flatten(tensor)

        # COO storage
        coo_sparse_storage = tensor_flatten.to_sparse_coo()
        ndim = coo_sparse_storage.indices().shape[0]
        nnz = coo_sparse_storage._nnz() # Number of non zero values
        # The computation of the number of bits is done as follows
        # -The first term corresponds to the number of bits necessary to store the indices.
        # The factor 32 corresponds to the number of bits necessary to store the int indices (int64 used in Pytorch according to doc but we can use small int as int32)
        # -The second term corresponds to the number of bits necessary to store the binary values -1 or 1 (which can be encoded using 2 bit).
        # -The third term corresponds to the number of bits necessary to store the floating point (32 bits) scaling coefficients
        # nb_bits = ndim*nnz*64 + nnz*2 + 32*2
        nb_bits = ndim*nnz*32 + nnz*2 + 32*2

    else:
        nb_bits = (tensor.element_size() * tensor.nelement())*8

    return nb_bits, torch.flatten(tensor).shape[0]

def get_nb_bits_model(exp_results_folder, is_model_ternarized):
    """
        Get the number of bits necessary to store the model

        Arguments:
        ----------
        model: torch model
            Model from which we want to compute the number of bits
        is_model_ternarized: bool
            True if the model has been ternarized
    """
    # Loading the parameters file to get the used model
    parameters_file = exp_results_folder + "/params_exp/params_beginning_0.pth"
    # Open the file
    with open(parameters_file, 'rb') as pf:
        params = pickle.load(pf)

    # Iterating over the trained models
    list_nb_bits_total_model = []
    list_nb_bits_quantized_layers_model = []
    nb_weights_to_quantize = 0
    nb_total_weights = 0
    for model_file in os.listdir(exp_results_folder + "/model/"):
        nb_bits_total_model = 0
        nb_bits_quantized_layers_model = 0
        if ('jit' not in model_file.lower()) and (os.path.isfile(exp_results_folder + "/model/" + model_file)) and ('chechkpoint' not in model_file.lower()):
            # Loading the model into memory
            model_dict = torch.load(exp_results_folder + "/model/" + model_file, map_location=torch.device('cpu'))
            model = model_dict['model']

            # Getting the list of the layers that have been quantized
            params_groups, quantized_layers = get_params_groups_to_quantize(model, params['model_to_use'])

            # Iterating over the layers
            for name, param in model.named_parameters():
                ternarized = False
                if (name in quantized_layers):
                    ternarized = True
                nb_bits_storage, nb_weights = nb_bits_store_tensor_coo(param, is_model_ternarized=is_model_ternarized, ternarized=ternarized)
                if (ternarized):
                    nb_bits_quantized_layers_model += nb_bits_storage
                    nb_weights_to_quantize += nb_weights
                nb_bits_total_model += nb_bits_storage
                nb_total_weights += nb_weights

            # Adding the results to the list of values
            list_nb_bits_total_model.append(nb_bits_total_model)
            list_nb_bits_quantized_layers_model.append(nb_bits_quantized_layers_model)

    print("\n\n\nNumber of total weights: {}".format(nb_total_weights))
    print("\tNumber of weights to quantize: {}".format(nb_weights_to_quantize))
    print("\tPercentage of weights that can be quantized: {}\n\n".format(100*nb_weights_to_quantize/nb_total_weights))

    return list_nb_bits_total_model, list_nb_bits_quantized_layers_model

def get_doReFa_nb_bits_storage(exp_results_folder):
    """
    Calculate the number of bits required to store the model (global and quantized layers only).

    Arguments:
    ----------
    exp_results_folder: str
        Path to the folder containing experimental results, including model and params files.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'nb_total_bits': Total bits required for the full model.
        - 'nb_quantized_bits': Total bits required for quantized layers.
        - 'nb_total_weights': Total number of weights in the model.
        - 'nb_quantized_weights': Number of weights in quantized layers.
        - 'quantized_percentage': Percentage of weights marked for quantization.
    """
    # Load the parameters file
    params_file = os.path.join(exp_results_folder, "params_exp/params_beginning_0.pth")
    with open(params_file, 'rb') as pf:
        params = pickle.load(pf)

    nb_total_bits = 0
    nb_quantized_bits = 0
    nb_total_weights = 0
    nb_quantized_weights = 0

    # Iterate over model files
    model_folder = os.path.join(exp_results_folder, "model/")
    for model_file in os.listdir(model_folder):
        if ('jit' not in model_file.lower()) and ('checkpoint' not in model_file.lower()) and os.path.isfile(os.path.join(model_folder, model_file)):
            # Load model
            model_dict = torch.load(os.path.join(model_folder, model_file), map_location=torch.device('cpu'))
            model = model_dict['model']

            # Get quantized layers
            params_groups, quantized_layers = get_params_groups_to_quantize(model, params['model_to_use'])

            # Process each parameter
            for name, param in model.named_parameters():
                if param.requires_grad:
                    num_weights = param.numel()
                    nb_total_weights += num_weights
                    nb_total_bits += 32 * num_weights  # Full precision: 32 bits per weight
                    
                    if name in quantized_layers:
                        # For quantized layers, count 1-bit weights + scaling factors (32 bits per layer)
                        nb_quantized_bits += num_weights + 32
                        nb_quantized_weights += num_weights

    quantized_percentage = 100 * nb_quantized_weights / nb_total_weights if nb_total_weights > 0 else 0

    return {
        'nb_total_bits': nb_total_bits,
        'nb_quantized_bits': nb_quantized_bits,
        'nb_total_weights': nb_total_weights,
        'nb_quantized_weights': nb_quantized_weights,
        'quantized_percentage': quantized_percentage
    }

def compute_doReFa_compression_gains(storage_info):
    """
    Compute global and local compression gains for the model.

    Arguments:
    ----------
    storage_info: dict
        Dictionary containing storage details:
        - 'nb_total_bits'
        - 'nb_quantized_bits'
        - 'nb_total_weights'
        - 'nb_quantized_weights'
        - 'quantized_percentage'

    Returns:
    --------
    dict
        Dictionary containing:
        - 'global_compression_gain': Compression gain for the full model.
        - 'local_compression_gain': Compression gain for quantized layers only.
    """
    print(storage_info)
    nb_total_bits = storage_info['nb_total_bits']
    nb_quantized_bits = storage_info['nb_quantized_bits']
    nb_quantized_weights = storage_info['nb_quantized_weights']
    nb_total_weights = storage_info['nb_total_weights']

    # Compute global compression gain
    nb_total_bits_after_compression = nb_total_bits - (32 * nb_quantized_weights) + nb_quantized_bits
    global_compression_gain = 100 - (nb_total_bits_after_compression / nb_total_bits * 100) if nb_total_bits > 0 else 0
    #global_compression_gain = 100 - (nb_quantized_bits / nb_total_bits * 100) if nb_total_bits > 0 else 0

    # Compute local compression gain
    if nb_quantized_weights > 0:
        local_compression_gain = 100 - ((nb_quantized_bits - (nb_total_bits - 32 * nb_total_weights)) / (32 * nb_quantized_weights) * 100)
    else:
        local_compression_gain = 0

    return {
        'global_compression_gain': global_compression_gain,
        'local_compression_gain': local_compression_gain
    }
def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--exp_results_folder', required=True, help="Path to the results folder of the experiment", type=str)
    ap.add_argument('--is_model_ternarized', required=True, help="True if the model has been ternarized", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    exp_results_folder = args['exp_results_folder']
    is_model_ternarized = args['is_model_ternarized']
    if (is_model_ternarized.lower() == "true"):
        is_model_ternarized = True
    else:
        is_model_ternarized = False

    #==========================================================================#
    # Getting the number of bits necessary for the experiment
    list_nb_bits_total_model, list_nb_bits_quantized_layers_model = get_nb_bits_model(exp_results_folder, is_model_ternarized)

    # Getting the mean values
    print("\n\n\nNumber of bits for the whole model: {} +- {} bits".format(np.mean(list_nb_bits_total_model), np.std(list_nb_bits_total_model)))
    print("\nNumber of bits for the quantized layers of model: {} +- {} bits \n".format(np.mean(list_nb_bits_quantized_layers_model), np.std(list_nb_bits_quantized_layers_model)))

if __name__=="__main__":
    main()
