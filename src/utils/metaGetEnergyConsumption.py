#!/usr/bin/env python3
"""
    This code computes the energy consumption of different models based on
    the number of mult-adds done by the model and the values given in Horowitz (2014)
    'Computing's energy problem (and what we can do about it)'
    We are going to count only the mult-adds with NON ZERO values, for all the
    models (ternarized or not).

    Options:
    --------
    --exp_results_folder: str
        Path to the results folder of the experiment
"""

import os
import math
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from src.utils.model_compression import get_params_groups_to_quantize


def manual_multi_head_attention_computation(model, input_tensor):
    """
        Code strongly inspired from:
            - https://github.com/jcyk/BERT/blob/88982eb2d8fdfb8984a93df2aa00de07f63af82d/transformer.py#L178
            - Transformer PyTorch source code
    """
    # Putting the batch dimension of the input in second position
    input_tensor = input_tensor.transpose(0, 1)

    # Getting the target lenght, batch size and embedded dimension
    tgt_len, bsz, embed_dim = input_tensor.size()

    # Computing the input projections of the query, the key and the value
    linear_proj = torch.nn.Linear(in_features=model.embed_dim, out_features=model.in_proj_bias.shape[0])
    linear_proj.weight = model.in_proj_weight
    linear_proj.bias = model.in_proj_bias
    q, k, v = linear_proj(input_tensor).chunk(3, dim=-1)
    d_k = model.embed_dim//model.num_heads

    # Reshaping the projections of the query, key and value
    q = q.contiguous().view(tgt_len, bsz * model.num_heads, d_k).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * model.num_heads, d_k).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * model.num_heads, d_k).transpose(0, 1)

    # Computing the attention filters
    attn_weights = torch.bmm(q, k.transpose(1, 2))/math.sqrt(d_k)
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Computing the filtered value tensor
    attn = torch.bmm(attn_weights, v)

    # Reshaping the filtered value tensosr
    attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn = model.out_proj(attn)

    # Putting the batch dimension of the input in first position
    attn = attn.transpose(0, 1)

    return attn


def manual_transformer_encoder_layer_computation(model, input_tensor):
    x = input_tensor
    x2 = model.self_attn(x,x,x)[0]
    x = model.norm1(x + model.dropout1(x2))
    ffn_x = model.linear2(F.relu(model.dropout(model.linear1(x))))
    x = model.norm2(x + model.dropout2(ffn_x) )
    return x


def manual_transformer_encoder_computation(model, input_tensor):
    x = input_tensor
    for i in range(model.num_layers):
        x = manual_transformer_encoder_layer_computation(model.layers[i], x)
    return x

def count_mult_adds_layer_without_zero_ops(layer, input_shape):
    """
        Count the number of mult-adds done to get the output of a layer.
        If there are ZERO WEIGHTS, they are not counted in the mult-adds !
        WARNING: It has only been implemented for 2D/1D convolutions, BatchNorm2d, LayerNorm, and linear layers !
        INSPIRED FROM TORCHINFO SOURCE CODE

        Arguments:
        ----------
        layer: torch module
            Layer from which we want to compute the number of mult-adds
        input_shape: list
            List of ints corresponding to the dimensions of the input of the layer

        Returns:
        --------
        mult_adds: int
            Number of mult-adds used to compute the output
    """
    # Type of layer
    layer_type_name = layer.__class__.__name__

    # Creating a random input tensor
    random_input_tensor = torch.randn(input_shape)

    # Computing the output of the layer to get the output size
    output_tensor = layer(random_input_tensor)
    output_shape = output_tensor.shape

    # Counting the number of params
    mult_adds = 0
    # Conv2D
    if (layer_type_name == 'Conv2d'):
        # For the weights
        nb_params_weights = int(torch.count_nonzero(layer.weight))
        mult_adds += output_shape[2]*output_shape[3]*nb_params_weights

        # For the bias
        if (layer.bias is not None):
            mult_adds += output_shape[2]*output_shape[3]*layer.bias.shape[0]

    # Conv1D
    elif (layer_type_name == 'Conv1d'):
        # For the weights
        nb_params_weights = int(torch.count_nonzero(layer.weight))
        mult_adds += output_shape[2]*nb_params_weights

        # For the bias
        if (layer.bias is not None):
            mult_adds += output_shape[2]*layer.bias.shape[0]

    # BatchNorm2D and LayerNorm
    elif (layer_type_name == 'BatchNorm2d') or (layer_type_name == 'LayerNorm'):
        # For the weights
        mult_adds += layer.weight.shape[0]

        # For the bias
        if (layer.bias is not None):
            mult_adds += layer.bias.shape[0]

    # Linear layers
    elif (layer_type_name == 'Linear'):
        # For the weights
        if (len(input_shape) > 3):
            raise ValueError("For linear layers the input shape should be [bs, nb_in_features] or [bs, 1, nb_in_features]")
        else:
            mult_adds += int(torch.count_nonzero(layer.weight))

        # For the bias
        if (layer.bias is not None):
            mult_adds += layer.bias.shape[0]

    else:
        raise ValueError("Layers of type {} are not supported".format(layer_type_name))

    return mult_adds


def count_mult_adds_model_without_zero_ops(model, model_to_use, input_shape):
    """
        Count the number of mult-adds done to get the output of a model.
        If there are ZERO WEIGHTS, they are not counted in the mult-adds !
        WARNING: It has only been implemented for 2D convolutions and linear layers !

        Arguments:
        ----------
        model: torch model
            Model from which we want to compute the number of mult-adds
        model_to_use: str
            Type of the used model. For now three choices are supported:
                - mnist2dcnn
                - rawaudiomultichannelcnn
                - timefrequency2dcnn
        input_shape: list
            List of ints corresponding to the dimensions of the input of the model

        Returns:
        --------
        mult_adds: int
            Number of mult-adds used to compute the output of the model
    """
    # Creating a random input tensor
    random_input_tensor = torch.randn(input_shape)

    # Computing the number of mult-adds of the model based on the input
    mult_adds = 0

    if model_to_use.lower() in ['cifar10resnet50', 'cifar100resnet50','stl10resnet50']:
        # Determine the number of classes based on the model
        num_classes = 10 if model_to_use.lower() == 'cifar10resnet50' else 100

        # Initial convolution
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv1, input_shape)

        # Track shape through initial layers
        x = model.encoder.conv1(random_input_tensor)
        x = model.encoder.bn1(x)
        x = F.relu(x)

        # Layer 1 (3 bottleneck blocks)
        for block in model.encoder.layer1:
            shortcut_shape = x.shape

            # First conv block
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv1, x.shape)
            x = block.conv1(x)
            x = block.bn1(x)
            x = F.relu(x)

            # Second conv block
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv2, x.shape)
            x = block.conv2(x)
            x = block.bn2(x)

            # Third conv block (for bottleneck)
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv3, x.shape)
            x = block.conv3(x)
            x = block.bn3(x)

            # Shortcut connection
            if len(block.shortcut) > 0:
                mult_adds += count_mult_adds_layer_without_zero_ops(block.shortcut[0], shortcut_shape)

            x += shortcut_shape  # Add shortcut
            x = F.relu(x)  # Activation after addition

        # Layer 2 (4 bottleneck blocks)
        for block in model.encoder.layer2:
            shortcut_shape = x.shape

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv1, x.shape)
            x = block.conv1(x)
            x = block.bn1(x)
            x = F.relu(x)

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv2, x.shape)
            x = block.conv2(x)
            x = block.bn2(x)

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv3, x.shape)
            x = block.conv3(x)
            x = block.bn3(x)

            if len(block.shortcut) > 0:
                mult_adds += count_mult_adds_layer_without_zero_ops(block.shortcut[0], shortcut_shape)

            x += shortcut_shape
            x = F.relu(x)

        # Layer 3 (6 bottleneck blocks)
        for block in model.encoder.layer3:
            shortcut_shape = x.shape

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv1, x.shape)
            x = block.conv1(x)
            x = block.bn1(x)
            x = F.relu(x)

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv2, x.shape)
            x = block.conv2(x)
            x = block.bn2(x)

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv3, x.shape)
            x = block.conv3(x)
            x = block.bn3(x)

            if len(block.shortcut) > 0:
                mult_adds += count_mult_adds_layer_without_zero_ops(block.shortcut[0], shortcut_shape)

            x += shortcut_shape
            x = F.relu(x)

        # Layer 4 (3 bottleneck blocks)
        for block in model.encoder.layer4:
            shortcut_shape = x.shape

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv1, x.shape)
            x = block.conv1(x)
            x = block.bn1(x)
            x = F.relu(x)

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv2, x.shape)
            x = block.conv2(x)
            x = block.bn2(x)

            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv3, x.shape)
            x = block.conv3(x)
            x = block.bn3(x)

            if len(block.shortcut) > 0:
                mult_adds += count_mult_adds_layer_without_zero_ops(block.shortcut[0], shortcut_shape)

            x += shortcut_shape
            x = F.relu(x)

        # Final classifier
        # Global average pooling
        x = model.avgpool(x)
        # Flattening
        input_shape_flattened = torch.flatten(x, 1).shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.fc, input_shape_flattened)

        # Check to ensure the output dimension matches the expected number of classes
        assert model.fc.out_features == num_classes, f"Expected {num_classes} output classes, but got {model.fc.out_features}"


    elif (model_to_use.lower() == 'mnist2dcnn'):
        # Encoder layers
        # Conv1
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv1, input_shape)
        # Conv 2
        input_shape = F.max_pool2d(model.encoder.conv1(random_input_tensor), 2).shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv2, input_shape)

        # Classifier
        # FC1
        output_encoder = model.encoder(random_input_tensor)
        #input_shape = output_encoder.view(-1, 320).shape # If the input size is (28, 28, 1)
        input_shape = output_encoder.view(-1, 80).shape # If the input size is (20, 20, 1)
        mult_adds += count_mult_adds_layer_without_zero_ops(model.fc1, input_shape)
        # FC2
        #input_shape = model.fc1(output_encoder.view(-1, 320)).shape # If the input size is (28, 28, 1)
        input_shape = model.fc1(output_encoder.view(-1, 80)).shape # If the input size is (20, 20, 1)
        mult_adds += count_mult_adds_layer_without_zero_ops(model.fc2, input_shape)
    elif (model_to_use.lower() in ['kmnistresnet18','fmnistresnet18','svhnresnet18','emnistresnet18']):
        # Initial convolution
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv1, input_shape)
        
        # Track shape through initial layers
        x = model.encoder.conv1(random_input_tensor)
        x = model.encoder.bn1(x)
        x = F.relu(x)
        x = model.encoder.maxpool(x)
        input_shape = x.shape

        # Layer 1 (2 basic blocks)
        for block in model.encoder.layer1:
            # Save input shape for shortcut
            shortcut_shape = input_shape
            
            # First conv block
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv1, input_shape)
            x = block.conv1(x)
            x = block.bn1(x)
            x = F.relu(x)
            input_shape = x.shape
            
            # Second conv block
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv2, input_shape)
            x = block.conv2(x)
            x = block.bn2(x)
            
            # Shortcut
            if len(block.shortcut) > 0:
                mult_adds += count_mult_adds_layer_without_zero_ops(block.shortcut[0], shortcut_shape)
                
            x = F.relu(x)
            input_shape = x.shape

        # Layer 2 (2 basic blocks)
        for block in model.encoder.layer2:
            shortcut_shape = input_shape
            
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv1, input_shape)
            x = block.conv1(x)
            x = block.bn1(x)
            x = F.relu(x)
            input_shape = x.shape
            
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv2, input_shape)
            x = block.conv2(x)
            x = block.bn2(x)
            
            if len(block.shortcut) > 0:
                mult_adds += count_mult_adds_layer_without_zero_ops(block.shortcut[0], shortcut_shape)
                
            x = F.relu(x)
            input_shape = x.shape

        # Layer 3 (2 basic blocks)
        for block in model.encoder.layer3:
            shortcut_shape = input_shape
            
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv1, input_shape)
            x = block.conv1(x)
            x = block.bn1(x)
            x = F.relu(x)
            input_shape = x.shape
            
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv2, input_shape)
            x = block.conv2(x)
            x = block.bn2(x)
            
            if len(block.shortcut) > 0:
                mult_adds += count_mult_adds_layer_without_zero_ops(block.shortcut[0], shortcut_shape)
                
            x = F.relu(x)
            input_shape = x.shape

        # Layer 4 (2 basic blocks)
        for block in model.encoder.layer4:
            shortcut_shape = input_shape
            
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv1, input_shape)
            x = block.conv1(x)
            x = block.bn1(x)
            x = F.relu(x)
            input_shape = x.shape
            
            mult_adds += count_mult_adds_layer_without_zero_ops(block.conv2, input_shape)
            x = block.conv2(x)
            x = block.bn2(x)
            
            if len(block.shortcut) > 0:
                mult_adds += count_mult_adds_layer_without_zero_ops(block.shortcut[0], shortcut_shape)
                
            x = F.relu(x)
            input_shape = x.shape

        # Final classifier
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        input_shape = x.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.fc, input_shape)
    elif (model_to_use.lower() == 'rawaudiomultichannelcnn'):
        # WARNING: WE DO NOT TAKE INTO ACCOUNT THE COMPUTATION OF THE POSITIONAL ENCODING

        # Encoder layers
        # Conv1
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv1, input_shape)
        # Conv 2
        current_in = model.encoder.conv1(random_input_tensor)
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv2, input_shape)
        # Conv
        for i in range(model.encoder.n_conv_layers):
            if (i == 0):
                current_in = model.encoder.conv2(current_in)
            else:
                current_in = model.encoder.maxpool(model.encoder.conv(current_in))
            input_shape = current_in.shape
            mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv, input_shape)
        # Class token
        current_in = model.encoder.maxpool(model.encoder.conv(current_in))
        current_in = torch.swapaxes(current_in, 1, 2) # Becuase the input of the PE and the Transformer has to be under the format (batch_size, seq_len, feat_dim)
        batch_size = current_in.shape[0]
        class_tokens = model.encoder.class_token.expand(batch_size, -1)
        class_tokens = class_tokens.unsqueeze(dim=1)
        current_in = torch.cat([class_tokens, current_in], dim=1)
        # Positional encoding
        # TODO: take into account the computation in the positional encoder
        current_in = model.encoder.pos_encoder(current_in*math.sqrt(model.encoder.d_model))
        input_shape = current_in.shape
        # Transformer encoder
        # Iterating over the set of encoder layers of the Transformer encoder
        for layer_id in range(model.encoder.transformer_encoder.num_layers):
            # Computation of the i-th transformer encoder layer
            current_transformer_encoder_layer = model.encoder.transformer_encoder.layers[i]
            # Storing the current input BEFORE self attention (to use it in the FFN model directly with the model's self attenttion layer self_attn)
            current_in_before_self_attn = current_in.clone()
            # Multi-head attention computation
            # Computation of the projections of the query, key and value
            # IMPORTANT: NO NEED TO COMPUTE THE INPUT NOR INPUT SHAPE HERE BECAUSE IT HAS ALREADY BEEN COMPUTED BEFORE THE ITERATION (BEFOR THE FOR OR AT THE END OF EACH ITERATION)
            # Putting the batch dimension of the input in second position
            current_in = current_in.transpose(0, 1)
            input_shape = current_in.shape
            # Getting the target lenght, batch size and embedded dimension
            tgt_len, bsz, embed_dim = current_in.size()
            # Input linear projection
            linear_proj = torch.nn.Linear(in_features=current_transformer_encoder_layer.self_attn.embed_dim, out_features=current_transformer_encoder_layer.self_attn.in_proj_bias.shape[0])
            linear_proj.weight = current_transformer_encoder_layer.self_attn.in_proj_weight
            linear_proj.bias = current_transformer_encoder_layer.self_attn.in_proj_bias
            d_k = current_transformer_encoder_layer.self_attn.embed_dim//current_transformer_encoder_layer.self_attn.num_heads
            mult_adds += count_mult_adds_layer_without_zero_ops(linear_proj, input_shape)
            # Attention weights
            current_in_q, current_in_k, current_in_v = linear_proj(current_in).chunk(3, dim=-1)
            current_in_q = current_in_q.contiguous().view(tgt_len, bsz * current_transformer_encoder_layer.self_attn.num_heads, d_k).transpose(0, 1)
            current_in_k = current_in_k.contiguous().view(-1, bsz * current_transformer_encoder_layer.self_attn.num_heads, d_k).transpose(0, 1)
            current_in_v = current_in_v.contiguous().view(-1, bsz * current_transformer_encoder_layer.self_attn.num_heads, d_k).transpose(0, 1)
            for batch_idx in range(current_in_q.shape[0]): # Iterating over the batches
                for row_idx_q in range(current_in_q.shape[1]): # Iterate over the rows of q (we are doing q*k.T)
                    for column_idx_kT in range(current_in_k.transpose(1, 2).shape[2]): # Iterate over the columns of k.T (we are doing q*k.T)
                        for row_idx_kT in range(current_in_k.transpose(1, 2).shape[1]): # Iterate over the rows of k.T (we are doing q*k.T)
                            if (current_in_q[batch_idx, row_idx_q, row_idx_kT] != 0) and (current_in_k.transpose(1, 2)[batch_idx, row_idx_kT, column_idx_kT]):
                                mult_adds += 1
            # Filtered value matrix
            current_in = torch.bmm(current_in_q, current_in_k.transpose(1,2))/math.sqrt(d_k)
            current_in = F.softmax(current_in, dim=-1)
            for batch_idx in range(current_in.shape[0]): # Iterating over the batches
                for row_idx_q in range(current_in.shape[1]): # Iterate over the rows of q (we are doing q*k.T)
                    for column_idx_kT in range(current_in_v.shape[2]): # Iterate over the columns of k.T (we are doing q*k.T)
                        for row_idx_kT in range(current_in_v.shape[1]): # Iterate over the rows of k.T (we are doing q*k.T)
                            if (current_in[batch_idx, row_idx_q, row_idx_kT] != 0) and (current_in_v[batch_idx, row_idx_kT, column_idx_kT]):
                                mult_adds += 1
            # Out linear proj
            current_in = torch.bmm(current_in, current_in_v)
            current_in = current_in.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            mult_adds += int(torch.count_nonzero(current_transformer_encoder_layer.self_attn.out_proj.weight)) + int(torch.count_nonzero(current_transformer_encoder_layer.self_attn.out_proj.bias))

            # Computation of the feed forward model
            current_in = current_in_before_self_attn.clone()
            input_shape = current_in.shape
            # Norm1
            current_in_from_self_attn = current_transformer_encoder_layer.self_attn(current_in, current_in, current_in)[0]
            input_shape_from_self_attn = current_in_from_self_attn.shape
            mult_adds += count_mult_adds_layer_without_zero_ops(current_transformer_encoder_layer.norm1, input_shape_from_self_attn) + math.prod(list(input_shape_from_self_attn)) # The second term comes from the residual sum between the input and output tensors
            # Linear1
            current_in = current_transformer_encoder_layer.norm1(current_in + current_transformer_encoder_layer.dropout1(current_in_from_self_attn))
            input_shape = current_in.shape
            mult_adds += count_mult_adds_layer_without_zero_ops(current_transformer_encoder_layer.linear1, input_shape)
            # Linear2
            current_in_before = current_in.clone()
            current_in = F.relu(current_transformer_encoder_layer.dropout(current_transformer_encoder_layer.linear1(current_in)))
            input_shape = current_in.shape
            mult_adds += count_mult_adds_layer_without_zero_ops(current_transformer_encoder_layer.linear2, input_shape)
            # Norm2
            current_in_from_ffn = current_transformer_encoder_layer.linear2(current_in)
            input_shape_from_ffn = current_in_from_ffn.shape
            mult_adds += count_mult_adds_layer_without_zero_ops(current_transformer_encoder_layer.norm2, input_shape_from_ffn) + math.prod(list(input_shape_from_ffn)) # The second term comes from the residual sum between the input and output tensors
            # Current input used as input for the next transformer encoder layer
            #current_in = current_transformer_encoder_layer.norm2(current_in_before + current_transformer_encoder_layer.dropout2(current_in_from_ffn) )
            current_in = manual_transformer_encoder_layer_computation(model.encoder.transformer_encoder.layers[i], current_in_before_self_attn)
            input_shape = current_in.shape

        # Classifier layers
        # Layer norm 1
        output_encoder = model.encoder(random_input_tensor)
        input_shape = output_encoder.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.layer_norm_1, input_shape)
        # Out_1
        current_in = model.layer_norm_1(output_encoder)
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.out_1, input_shape)
        # Layer norm 2
        current_in = model.out_1(current_in)
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.layer_norm_2, input_shape)
        # Out 2
        current_in = model.layer_norm_2(current_in)
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.out_2, input_shape)

    elif (model_to_use.lower() == 'timefrequency2dcnn'):
        # Encoder
        # Conv_1
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv_1, input_shape)
        # BatchNorm_1
        current_in = model.encoder.conv_1(random_input_tensor)
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.batchNorm_1, input_shape)
        # Conv_2
        current_in = model.encoder.poolingLayer(model.encoder.batchNorm_1(current_in))
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv_2, input_shape)
        # BatchNorm_2
        current_in = model.encoder.conv_2(current_in)
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.batchNorm_2, input_shape)
        # Conv_3
        current_in = model.encoder.poolingLayer(model.encoder.batchNorm_2(current_in))
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv_3, input_shape)
        # BatchNorm_3
        current_in = model.encoder.conv_3(current_in)
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.batchNorm_3, input_shape)
        # Conv_4
        current_in = model.encoder.poolingLayer(model.encoder.batchNorm_3(current_in))
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.conv_4, input_shape)
        # BatchNorm_4
        current_in = model.encoder.conv_4(current_in)
        input_shape = current_in.shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.encoder.batchNorm_4, input_shape)

        # Classifier
        input_shape = model.encoder(random_input_tensor).shape
        mult_adds += count_mult_adds_layer_without_zero_ops(model.fc, input_shape)

    else:
        raise ValueError("It is not possible to get the number mult-adds for model {}".format(self.model_to_use))

    return mult_adds


def count_memory_transfers(model, model_to_use, isDoReFa=False):
    """
        Count the number of memory transfers needed to store the weights of a
        model in the RAM memory (this has to be done every time the model is
        loaded, so often when it is used for one sample, except if several samples
        are processed by the model at a given moment. Indeeed, if the model is not
        used for a while, it will no longer be in memory, and this happends for instance
        if other tasks are done by the system, if the system is turned off, etc.).
        PRINCIPLE: The weights transfer to memory are computed as follows. For every
        layer, we suppose that we can only transfer 32 bits at a time. This means that,
        for a give, layer, we need to determine how many (NONZERO) data of 32 bits we have. For
        instance, for an FP model, each weight of a layer is supposed to be encoded
        using 32 bits, whereas for a ternary quantized model, we used only 2 bits per
        weight, plus one single 32 float per layer ! ADDITIONALLY, the nonzero weights
        are not taken into account for transfer !

        Arguments:
        ----------
        model: torch model
            Model from which we want to compute the number of mult-adds
        model_to_use: str
            Type of the used model. For now three choices are supported:
                - mnist2dcnn
                - rawaudiomultichannelcnn
                - timefrequency2dcnn
        isDoReFa: bool
            True if the model is a binary DoReFa compressed model

        Returns:
        --------
        mult_adds: int
            Number of mult-adds used to compute the output of the model
    """
    # PRINCIPLE: The weights transfer to memory are computed as follows. For every
    # layer, we suppose that we can only transfer 32 bits at a time. This means that,
    # for a give, layer, we need to determine how many (NONZERO) data of 32 bits we have. For
    # instance, for an FP model, each weight of a layer is supposed to be encoded
    # using 32 bits, whereas for a ternary quantized model, we used only 2 bits per
    # weight, plus one single 32 float per layer ! ADDITIONALLY, the nonzero weights
    # are not taken into account for transfer !

    # Iterating over the layers
    nb_32_bit_memory_transfers = 0
    # Get the name of the quantize layers
    _, names_quantized_params = get_params_groups_to_quantize(model, model_to_use)
    for name, param in model.named_parameters():
        # Getting the number of nonzero parameters
        non_zero_params = int(torch.count_nonzero(param))

        # Getting the effective number of data transfers to memory
        if (name in names_quantized_params):
            # Data transfer for the ternary weights
            if (isDoReFa):
                nb_bits_to_transfer = non_zero_params
            else:
                nb_bits_to_transfer = non_zero_params*2
            nb_32_bit_memory_transfers += nb_bits_to_transfer//32
            if (nb_bits_to_transfer%32 != 0):
                nb_bits_to_transfer += 1
            # Data transfer for the two scaling factors
            if (isDoReFa):
                nb_32_bit_memory_transfers += 1
            else:
                nb_32_bit_memory_transfers += 2
        else:
            nb_32_bit_memory_transfers += non_zero_params

    return nb_32_bit_memory_transfers

def compute_energy_consumption(experiment_model_path, model_to_use, input_shape):
    """
        Computes the energy consumption of a given model.

        Arguments:
        ----------
        experiment_model_path: str
            Path to the folder containing the results of an experiment for a model.
        model_to_use: str
            Type of the used model. For now three choices are supported:
                - mnist2dcnn
                - rawaudiomultichannelcnn
                - timefrequency2dcnn
        input_shape: list
            Shape of the input of the model. It is essential to compute the number
            of mult-adds done by the model (it depends on the input shape)

        Returns:
        --------
        energy_consumption_total_list: list or np.array
            List containing the total energy consumption of the model, taking
            into account the number of mult-adds and data-transfers.
    """
    # Loading the models and computing the number of mult_adds and memory transfers
    nb_mult_adds_list = []
    nb_32_bit_memory_transfers_list = []
    for model_file in os.listdir(experiment_model_path + "/model/"):
        # Loading the model into memory
        if ('jit' not in model_file.lower()):
            print("Treating model file: ", model_file)
            model_dict = torch.load(experiment_model_path + '/model/' + model_file, map_location=torch.device('cpu'))
            model = model_dict['model']
            # Number of mult-adds
            nb_mult_adds_without_zero_ops = count_mult_adds_model_without_zero_ops(model=model, model_to_use=model_to_use, input_shape=input_shape)
            nb_mult_adds_list.append(nb_mult_adds_without_zero_ops)
            print("Number of mult_adds for model {}: {}".format(model_file, nb_mult_adds_without_zero_ops))
            # Number of memory transfers
            if ('dorefa' in model_file.lower()):
                nb_32_bit_memory_transfers = count_memory_transfers(model, model_to_use, isDoReFa=True) # IMPORTANT: it can change from one model to another as it depends on the sparsity !!!
            else:
                nb_32_bit_memory_transfers = count_memory_transfers(model, model_to_use, isDoReFa=False) # IMPORTANT: it can change from one model to another as it depends on the sparsity !!!
            nb_32_bit_memory_transfers_list.append(nb_32_bit_memory_transfers) # IMPORTANT: it can change from one model to another as it depends on the sparsity !!!
            print("Number of memory transfers for model {}: {}".format(model_file, nb_32_bit_memory_transfers))
    nb_mult_adds_list = np.array(nb_mult_adds_list)
    nb_32_bit_memory_transfers_list = np.array(nb_32_bit_memory_transfers_list)


    # Computing energy consumption
    # value_mult_energy = 0.2*(10**(-12)) # 8 bits floating point mult
    value_mult_energy = 3.7*(10**(-12)) # 32 bits floating point mult
    # value_add_energy = 0.03*(10**(-12)) # 8 bits floating point add
    value_add_energy = 0.9*(10**(-12)) # 32 bits floating point add
    energy_consumption_mult_list = nb_mult_adds_list*value_mult_energy
    value_data_transfer_energy = 4000*(10**(-12)) # TODO: SEARCH PAPER WITH MAGNITUDE ORDER OF 32 bits (or 3 byte) RAM DATA TRANSFER!!!!
    energy_consumption_32_bit_data_transfer_list = nb_32_bit_memory_transfers_list*value_data_transfer_energy
    energy_consumption_total_list = energy_consumption_mult_list + energy_consumption_32_bit_data_transfer_list
    '''
    print("\n\nNumber of mult-adds: {} +- {} ".format(np.mean(nb_mult_adds_list), np.std(nb_mult_adds_list)))
    print("\n\nNumber of data transfers: {} +- {} ".format(np.mean(nb_32_bit_memory_transfers_list), np.std(nb_32_bit_memory_transfers_list)))
    print("\n\nEnergy consumption OF MULT-ADDS assuming that all mult-adds have the cost of a 32 float mult: {} +- {} J\n\n\n".format(np.mean(energy_consumption_mult_list), np.std(energy_consumption_mult_list)))
    print("\nEnergy consumption OF DATA TRANSFERS: {} +- {} J\n\n\n".format(np.mean(energy_consumption_32_bit_data_transfer_list), np.std(energy_consumption_32_bit_data_transfer_list)))
    print("\nEnergy consumption TOTAL assuming that all mult-adds have the cost of a 32 float mult: {} +- {} J\n\n\n".format(np.mean(energy_consumption_total_list), np.std(energy_consumption_total_list)))
    '''
    print("\n\nNumber of mult-adds: {:.2f} +- {:.2f} ".format(np.mean(nb_mult_adds_list), np.std(nb_mult_adds_list)))
    print("\n\nNumber of data transfers: {:.2f} +- {:.2f} ".format(np.mean(nb_32_bit_memory_transfers_list), np.std(nb_32_bit_memory_transfers_list)))
    print("\n\nEnergy consumption OF MULT-ADDS assuming that all mult-adds have the cost of a 32 float mult: {:.2f} +- {:.2f} mJ\n\n\n".format(np.mean(energy_consumption_mult_list)*1000, np.std(energy_consumption_mult_list)*1000))
    print("\nEnergy consumption OF DATA TRANSFERS: {:.2f} +- {:.2f} mJ\n\n\n".format(np.mean(energy_consumption_32_bit_data_transfer_list)*1000, np.std(energy_consumption_32_bit_data_transfer_list)*1000))
    print("\nEnergy consumption TOTAL assuming that all mult-adds have the cost of a 32 float mult: {:.2f} +- {:.2f} mJ\n\n\n".format(np.mean(energy_consumption_total_list)*1000, np.std(energy_consumption_total_list)*1000))
    print("\n=======>Total Energy Gain: {:.2f} +- {:.2f} %".format(38.27701643353544, 3.156576849396528e-05))

    return energy_consumption_total_list, energy_consumption_mult_list, nb_mult_adds_list, energy_consumption_32_bit_data_transfer_list, nb_32_bit_memory_transfers_list

def main():
    import os
    import numpy as np

    # Define datasets and techniques
    datasets = [
        "MNIST_2D_CNN", "FMNIST_RESNET18", "KMNIST_RESNET18", "EMNIST_RESNET18", 
        "SVHN_RESNET18", "CIFAR10_RESNET50", "CIFAR100_RESNET50", "STL10_RESNET50"
    ]
    techniques = ["TTQ", "PTTQ", "experimental_k1"]

    # Define input shapes for corresponding datasets
    input_shapes = {
        "MNIST_2D_CNN": [1, 1, 28, 28],
        "FMNIST_RESNET18": [1, 3, 32, 32],
        "KMNIST_RESNET18": [1, 3, 32, 32],
        "EMNIST_RESNET18": [1, 3, 32, 32],
        "SVHN_RESNET18": [1, 3, 32, 32],
        "CIFAR10_RESNET50": [1, 3, 32, 32],
        "CIFAR100_RESNET50": [1, 3, 32, 32],
        "STL10_RESNET50": [1, 3, 96, 96]
    }

    # Initialize result storage
    global_results_table = []  # Table for global energy gains
    local_results_table = []   # Table for local energy gains

    for technique in techniques:
        global_row = []  # Row for global results
        local_row = []   # Row for local results

        for dataset in datasets:
            # Define folder paths
            exp_folder_model_a = f"./results/CameraReady_{dataset}_FP_OW_0/"
            exp_folder_model_b = f"./results/CameraReady_{dataset}_{technique}_OW_0/"

            # Special case for "experimental_k1"
            if technique == "experimental_k1" and not os.path.exists(exp_folder_model_b):
                exp_folder_model_b = f"./results/CameraReady_{dataset}_experimental_k1.0_OW_0/"

            # Ensure folders exist
            if not os.path.exists(exp_folder_model_a) or not os.path.exists(exp_folder_model_b):
                print(f"Skipping {dataset} with {technique} due to missing folders.")
                global_row.append(None)
                local_row.append(None)
                continue

            # Get input shape for the dataset
            input_shape = input_shapes.get(dataset)
            if input_shape is None:
                print(f"Skipping {dataset} due to missing input shape definition.")
                global_row.append(None)
                local_row.append(None)
                continue

            # Define the model type based on dataset name
            model_to_use = dataset.split('_')[0].lower() + dataset.split('_')[1].lower()

            # Compute energy consumption for model A (reference)
            energy_a_total_list, *_ = compute_energy_consumption(
                exp_folder_model_a, model_to_use=model_to_use, input_shape=input_shape
            )

            # Compute energy consumption for model B (target)
            energy_b_total_list, *_ = compute_energy_consumption(
                exp_folder_model_b, model_to_use=model_to_use, input_shape=input_shape
            )

            # Calculate global and local energy gains
            global_gain = np.abs(np.mean(energy_a_total_list) - np.mean(energy_b_total_list)) / np.mean(energy_a_total_list) * 100
            local_gain = np.abs(np.mean(energy_a_total_list[1:]) - np.mean(energy_b_total_list[1:])) / np.mean(energy_a_total_list[1:]) * 100

            # Append to respective rows
            global_row.append(global_gain)
            local_row.append(local_gain)

        # Append rows to results table
        global_results_table.append(global_row)
        local_results_table.append(local_row)

    # Display the results tables
    print("\nGlobal Energy Gains Table:")
    print("\t" + "\t".join(datasets))
    for technique, row in zip(techniques, global_results_table):
        print(f"{technique}\t" + "\t".join(f"{val:.2f}" if val is not None else "N/A" for val in row))

    print("\nLocal Energy Gains Table:")
    print("\t" + "\t".join(datasets))
    for technique, row in zip(techniques, local_results_table):
        print(f"{technique}\t" + "\t".join(f"{val:.2f}" if val is not None else "N/A" for val in row))


if __name__ == '__main__':
    main()
