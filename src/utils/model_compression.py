#!/usr/bin/env python3
"""
    Implementation of some functions useful for model compression using ternary
    quantization
"""
import torch

#==============================================================================#
#====Functions for Ternary Networks from the paper of Heinrich et al. (2018)====#
#==============================================================================#
# ternary weight approximation according to https://arxiv.org/abs/1605.04711
def approx_weights(w_in):
    """
        Function from https://github.com/mattiaspaul/TernaryNet/blob/master/ternaryNet_github.py
    """
    a,b,c,d = w_in.size()
    delta = 0.7*torch.mean(torch.mean(torch.mean(torch.abs(w_in),dim=3),dim=2),dim=1).view(-1,1,1,1)
    alpha = torch.abs(w_in)*(torch.abs(w_in)>delta).float()
    alpha = (torch.sum(torch.sum(torch.sum(alpha,dim=3),dim=2),dim=1)  \
    /torch.sum(torch.sum(torch.sum((alpha>0).float(),dim=3),dim=2),dim=1)).view(-1,1,1,1)
    w_out = -(w_in<-delta).float()*alpha + (w_in>delta).float()*alpha
    return w_out

# ternary weight approximation for FC layers
def approx_weights_fc(w_in):
    delta = 0.7*torch.mean(torch.abs(w_in),dim=1).view(-1,1)
    alpha = torch.abs(w_in)*(torch.abs(w_in)>delta).float()
    alpha = (torch.sum(alpha,dim=1)  \
    /torch.sum((alpha>0).float(),dim=1)).view(-1,1)
    w_out = -(w_in<-delta).float()*alpha + (w_in>delta).float()*alpha
    return w_out


#==============================================================================#
#=====Functions for asymmetric ternary quantization from Zhu et al. (2017)=====#
#==============================================================================#
def quantize(kernel, w_p, w_n, t):
    """
    Function from: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py

    Return quantized weights of a layer.
    Only possible values of quantized weights are: {zero, w_p, -w_n}.
    """
    delta = t*kernel.abs().max()
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    return w_p*a + (-w_n*b)

def quantize_two_thresh(kernel, w_r, w_l, x, y):
    """
    Function based on: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py
    ATTENTION: it is not the same function as we change the method to quantize
    the weights.

    Return quantized weights of a layer.
    Only possible values of quantized weights are: {zero, w_l, w_r}.
    """
    delta_min = kernel.mean() + x*kernel.std()
    delta_max = kernel.mean() + y*kernel.std()
    a = (kernel > delta_max).float()
    b = (kernel < delta_min).float()
    return w_r*a + w_l*b


def get_grads(kernel_grad, kernel, w_p, w_n, t):
    """
    Function from: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py

    Arguments:
        kernel_grad: gradient with respect to quantized kernel.
        kernel: corresponding full precision kernel.
        w_p, w_n: scaling factors.
        t: hyperparameter for quantization.
    Returns:
        1. gradient for the full precision kernel.
        2. gradient for w_p.
        3. gradient for w_n.
    """
    delta = t*kernel.abs().max()
    # masks
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    c = torch.ones(kernel.size()).cuda() - a - b
    # scaled kernel grad and grads for scaling factors (w_p, w_n)
    return w_p*a*kernel_grad + w_n*b*kernel_grad + 1.0*c*kernel_grad,\
        (a*kernel_grad).sum(), (b*kernel_grad).sum()

def get_grads_two_thresh(kernel_grad, kernel, w_r, w_l, x, y):
    """
    Function from: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py
    ATTENTION: it is not the same function as we change the method to quantize
    the weights.

    Arguments:
        kernel_grad: gradient with respect to quantized kernel.
        kernel: corresponding full precision kernel.
        w_r, w_l: scaling factors.
        x, y: hyperparameter for quantization.
    Returns:
        1. gradient for the full precision kernel.
        2. gradient for w_r.
        3. gradient for w_l.
    """
    delta_min = kernel.mean() + x*kernel.std()
    delta_max = kernel.mean() + y*kernel.std()
    # masks
    a = (kernel > delta_max).float()
    b = (kernel < delta_min).float()
    c = torch.ones(kernel.size()).cuda() - a - b
    # scaled kernel grad and grads for scaling factors (w_p, w_n)
    return w_r*a*kernel_grad + w_l*b*kernel_grad + 1.0*c*kernel_grad,\
        (a*kernel_grad).sum(), (b*kernel_grad).sum()


def get_params_groups_to_quantize(model, model_to_use):
    """
        Get the groups of the parameters to quantize.

        Arguments:
        ----------
        model: torch model
            Torch model from which we want to get the parameters to quantize
        model_to_use: str
            Type of the model to use. Three choices: mnist2dcnn, rawaudiomultichannelcnn,
            and timefrequency2dcnn

        Returns:
        --------

    """
    #======================================================================#
    #================================2D CNN================================#
    #======================================================================#
    names_params_to_be_quantized = []
    if (model_to_use.lower() in ['mnist2dcnn','fmnist2dcnn']):
        # Last FC layer
        weights_last_fc = [model.fc2.weight]

        # Parameters to quantize
        # Only the convolutions
        weights_to_be_quantized = [p for n, p in model.named_parameters() if ('conv' in n) and ('bias' not in n)]
        names_params_to_be_quantized = [n for n, p in model.named_parameters() if ('conv' in n) and ('bias' not in n)]

        # Parameters of batch_norm layers
        bn_weights = [p for n, p in model.named_parameters() if 'norm' in n and 'weight' in n]

        # Biases
        biases = [p for n, p in model.named_parameters() if 'bias' in n]

        params = {
                    'LastFCLayer': {'params': weights_last_fc},
                    'ToQuantize': {'params': weights_to_be_quantized},
                    'BNWeights': {'params': bn_weights},
                    'Biases': {'params': biases}
                 }

    elif model_to_use.lower() in ['mnistvit']:
        # Last FC layer (classification head)
        weights_last_fc = [model.fc.weight]
        
        # Parameters to quantize - explicitly exclude norm layers and only get weights
        weights_to_be_quantized = [p for n, p in model.named_parameters() 
                                  if ('transformer_encoder' in n) 
                                  and ('norm' not in n)  # Exclude norm layers
                                  and ('weight' in n) 
                                  and ('bias' not in n)
                                  #and ('linear' not in n)                      # For testing purposes only
                                  ]
        
        names_params_to_be_quantized = [n for n, p in model.named_parameters() 
                                       if ('transformer_encoder' in n) 
                                       and ('norm' not in n)  # Exclude norm layers
                                       and ('weight' in n) 
                                       and ('bias' not in n)]
        
        # Layer normalization weights - explicitly get only norm layer weights

        for n,p in model.named_parameters():
            print(f"Named PARAM : ", n)

        ln_weights = [p for n, p in model.named_parameters()
                      if ('transformer_encoder' in n)
                      and ('norm' in n) 
                      and ('weight' in n)]
        
        # All bias terms - explicitly exclude those already included
        biases = [p for n, p in model.named_parameters()
                  if ('bias' in n)]
        
        # Verification step - ensure no parameter appears in multiple groups
        all_params = set()
        for group_params in [weights_last_fc, weights_to_be_quantized, ln_weights, biases]:
            for param in group_params:
                param_id = id(param)
                if param_id in all_params:
                    print(f"Duplicate parameter found: {param}")
                all_params.add(param_id)
        
        params = {
            'LastFCLayer': {'params': weights_last_fc},
            'ToQuantize': {'params': weights_to_be_quantized},
            'LNWeights': {'params': ln_weights},
            'Biases': {'params': biases}
        }
        #params = {} # REVERT
    elif (model_to_use.lower() in ['kmnistresnet18', 'emnistresnet18','fmnistresnet18','svhnresnet18']):
        # Last FC layer (for ResNet18, this is the final fc layer)
        weights_last_fc = [model.fc.weight]
        
        # Parameters to quantize
        # All convolution layers including those in BasicBlocks
        weights_to_be_quantized = [p for n, p in model.named_parameters() 
                                  if ('conv' in n) and ('bias' not in n)]
        names_params_to_be_quantized = [n for n, p in model.named_parameters() 
                                       if ('conv' in n) and ('bias' not in n)]
        
        # Parameters of batch_norm layers (including those in BasicBlocks)
        bn_weights = [p for n, p in model.named_parameters() 
                      if ('bn' in n or 'batch_norm' in n) and ('weight' in n)]
        
        # All bias terms (including those in conv, bn, and fc layers)
        biases = [p for n, p in model.named_parameters() 
                  if 'bias' in n]
        
        params = {
            'LastFCLayer': {'params': weights_last_fc},
            'ToQuantize': {'params': weights_to_be_quantized},
            'BNWeights': {'params': bn_weights},
            'Biases': {'params': biases}
        }
    elif (model_to_use.lower() in ['kmnistdensenet', 'emnistdensenet', 'fmnistdensenet', 'svhndensenet']):
        # Last FC layer (for DenseNet, it's the "classifier" attribute in the model)
        weights_last_fc = [model.classifier.weight]

        # Parameters to quantize: all convolution layers, but not their bias terms
        weights_to_be_quantized = [
            p for n, p in model.named_parameters()
            if ('conv' in n) and ('bias' not in n)
        ]
        names_params_to_be_quantized = [
            n for n, p in model.named_parameters()
            if ('conv' in n) and ('bias' not in n)
        ]

        # BatchNorm weights (including those in DenseBlocks, transitions, etc.)
        bn_weights = [
            p for n, p in model.named_parameters()
            if (('bn' in n) or ('batch_norm' in n)) and ('weight' in n)
        ]

        # All bias terms (conv, bn, classifier, etc.)
        biases = [
            p for n, p in model.named_parameters()
            if 'bias' in n
        ]

        params = {
            'LastFCLayer': {'params': weights_last_fc},
            'ToQuantize': {'params': weights_to_be_quantized},
            'BNWeights': {'params': bn_weights},
            'Biases': {'params': biases},
        }

    elif (model_to_use.lower() in ['fmnistenet']):
        # Last FC layer
        weights_last_fc = [model.fc.weight]

        # All convolution layers (including conv_stem, conv_head, and those in MBConvBlocks)
        weights_to_be_quantized = [
            p for n, p in model.named_parameters()
            if ('conv' in n) and ('bias' not in n)
        ]
        names_params_to_be_quantized = [
            n for n, p in model.named_parameters()
            if ('conv' in n) and ('bias' not in n)
        ]

        # Parameters of batch_norm layers (including conv_stem, conv_head, and MBConvBlocks)
        bn_weights = [
            p for n, p in model.named_parameters()
            if (('bn' in n) or ('batch_norm' in n)) and ('weight' in n)
        ]

        # All bias terms
        biases = [
            p for n, p in model.named_parameters()
            if 'bias' in n
        ]

        # Group parameters into a dictionary for easy reference/optimization
        params = {
            'LastFCLayer': {'params': weights_last_fc},
            'ToQuantize': {'params': weights_to_be_quantized},
            'BNWeights': {'params': bn_weights},
            'Biases': {'params': biases}
        }

    elif (model_to_use.lower() in ['fmnistinceptionv4']):
        # -----------------------------------------------------
        # 1) Last FC layer (adjust this if your model's final 
        #    fully-connected layer has a different name)
        # -----------------------------------------------------
        if hasattr(model, 'fc') and model.fc is not None:
            weights_last_fc = [model.fc.weight]
        else:
            # Fallback: if your final layer is named differently, 
            # adjust here (e.g. model.classifier, model.last_linear, etc.)
            weights_last_fc = []
        
        # -----------------------------------------------------
        # 2) All convolution layers' weights 
        #    (any parameter name containing ".conv" but not ".bias")
        # -----------------------------------------------------
        weights_to_be_quantized = [
            p for n, p in model.named_parameters()
            if ('conv' in n) and ('bias' not in n)
        ]
        names_params_to_be_quantized = [
            n for n, p in model.named_parameters()
            if ('conv' in n) and ('bias' not in n)
        ]
        
        # -----------------------------------------------------
        # 3) All batch-norm layers’ weights 
        #    (any parameter name containing ".bn" or ".batch_norm",
        #     and which also contains "weight")
        # -----------------------------------------------------
        bn_weights = [
            p for n, p in model.named_parameters()
            if (('bn' in n) or ('batch_norm' in n)) and ('weight' in n)
        ]
        
        # -----------------------------------------------------
        # 4) All bias terms
        # -----------------------------------------------------
        biases = [
            p for n, p in model.named_parameters()
            if 'bias' in n
        ]
        
        # -----------------------------------------------------
        # Group parameters into a dictionary for easy reference 
        # and/or separate LR / optimization steps
        # -----------------------------------------------------
        params = {
            'LastFCLayer': {'params': weights_last_fc},
            'ToQuantize': {'params': weights_to_be_quantized},
            'BNWeights': {'params': bn_weights},
            'Biases': {'params': biases}
        }
    elif (model_to_use.lower() in ['cifar10resnet50','cifar100resnet50','stl10resnet50']):
        # Last FC layer
        weights_last_fc = [model.fc.weight]

        # Parameters to quantize
        # All convolution layers including those in Bottleneck blocks
        weights_to_be_quantized = [p for n, p in model.named_parameters() if ('conv' in n) and ('bias' not in n)]
        names_params_to_be_quantized = [n for n, p in model.named_parameters() if ('conv' in n) and ('bias' not in n)]

        # Parameters of batch_norm layers (including those in Bottleneck blocks)
        bn_weights = [p for n, p in model.named_parameters() if ('bn' in n or 'batch_norm' in n) and ('weight' in n)]

        # All bias terms (including those in conv, bn, and fc layers)
        biases = [p for n, p in model.named_parameters() if 'bias' in n]

        params = {
            'LastFCLayer': {'params': weights_last_fc},
            'ToQuantize': {'params': weights_to_be_quantized},
            'BNWeights': {'params': bn_weights},
            'Biases': {'params': biases}
        }
    #======================================================================#
    #==========================1D CNN-Transformer==========================#
    #======================================================================#
    elif (model_to_use.lower() == 'rawaudiomultichannelcnn'):
        # Separation of the different parameters
        transformer_params = []
        weights_to_be_quantized = []
        bn_weights = []
        biases = []
        for n, p in model.named_parameters():
            # Boolean to see if the parameter has been already associated to a group
            associated_param_to_group = False

            # Parameters to quantize
            # Convolution 2 and transformer layers
            if (('conv2' in n) and ('bias' not in n)) or ('transformer' in n and 'linear2.weight' in n):
                weights_to_be_quantized.append(p)
                names_params_to_be_quantized.append(n)
                associated_param_to_group = True

            # Parameters of batch_norm layers
            if ('norm' in n) and ('weight' in n):
                bn_weights.append(p)
                associated_param_to_group = True

            # Biases
            if ('bias' in n):
                biases.append(p)
                associated_param_to_group = True

            # Transformer parameters
            # Convolutions and transformer layers
            if (not associated_param_to_group):
                transformer_params.append(p)

        params = {
                    'Transformer': {'params': transformer_params},
                    'ToQuantize': {'params': weights_to_be_quantized},
                    'BNWeights': {'params': bn_weights},
                    'Biases': {'params': biases}
                 }

    #======================================================================#
    #=============================2D CNN HITS =============================#
    #======================================================================#
    elif (model_to_use.lower() == 'timefrequency2dcnn'):
        # Separation of the different parameters
        other_params = []
        weights_to_be_quantized = []
        bn_weights = []
        biases = []
        for n, p in model.named_parameters():
            # Boolean to see if the parameter has been already associated to a group
            associated_param_to_group = False

            # Parameters to quantize
            # Convolutions except the first one
            if ('conv' in n and 'conv_1' not in n) and ('bias' not in n):
                weights_to_be_quantized.append(p)
                names_params_to_be_quantized.append(n)
                associated_param_to_group = True

            # Parameters of batch_norm layers
            if ('Norm' in n) and ('weight' in n):
                bn_weights.append(p)
                associated_param_to_group = True

            # Biases
            if ('bias' in n):
                biases.append(p)
                associated_param_to_group = True

            # Other params
            if (not associated_param_to_group):
                other_params.append(p)

        params = {
                    'OtherParams': {'params': other_params},
                    'ToQuantize': {'params': weights_to_be_quantized},
                    'BNWeights': {'params': bn_weights},
                    'Biases': {'params': biases}
                 }

    #======================================================================#
    #============================ Other models ============================#
    #======================================================================#
    else:
        raise ValueError("Model to use {} is not valid for quantization".format(model_to_use))

    return params, names_params_to_be_quantized


def pruning_function_pTTQ(x, alpha, t_min, t_max):
    """
        Function inspired from the work of Manessi et al. (2019)
        Compute a pruning function of the input tensor x
        based on two threshold depending on the weight statistics, at a "speed" alpha.
        WARNING: there is not actual pruning that is done, but the value of x is
        set very close to zero if it is in an interval defined by the thresholds.
        IMPORTANT: WE MAKE THE HYPOTHESIS THAT THE WEIGHTS MEAN IS RELATIVELY CLOSE
        TO ZERO, AND THAT WE HAVE TWO THRESHOLDS, ONE FOR THE POSITIVE AND ONE
        FOR THE NEGATIVE WEIGHTS.

        Arguments:
        ----------
        x: torch.tensor
            Tensor to 'prune'
        alpha: float
            Hyper-parameter defining the 'speed' of the pruning.
        t_min: float
            Real (positive or negative) parameter used to compute the threshold
            parameter of the pruning based on the weights statistics.
        t_max: float
            Real (positive or negative) parameter used to compute the threshold
            parameter of the pruning based on the weights statistics.
    """
    # Defining the ReLU and Sigmoid functions
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Defining the thresholds
    x_mean, x_std = x.mean(), x.std()
    delta_min = (x_mean + t_min*x_std).abs()
    delta_max = (x_mean + t_max*x_std).abs()

    # Computing the output
    res = relu(x-delta_max)+delta_max*sigmoid(alpha*(x-delta_max)) - relu(-x-delta_min)-delta_min*sigmoid(alpha*(-x-delta_min))

    return res

def pruning_function_pTTQ_track(x, alpha, t_min, t_max):
    """
        Function inspired from the work of Manessi et al. (2019)
        Compute a pruning function of the input tensor x
        based on two threshold depending on the weight statistics, at a "speed" alpha.
        WARNING: there is not actual pruning that is done, but the value of x is
        set very close to zero if it is in an interval defined by the thresholds.
        IMPORTANT: WE MAKE THE HYPOTHESIS THAT THE WEIGHTS MEAN IS RELATIVELY CLOSE
        TO ZERO, AND THAT WE HAVE TWO THRESHOLDS, ONE FOR THE POSITIVE AND ONE
        FOR THE NEGATIVE WEIGHTS.

        Arguments:
        ----------
        x: torch.tensor
            Tensor to 'prune'
        alpha: float
            Hyper-parameter defining the 'speed' of the pruning.
        t_min: float
            Real (positive or negative) parameter used to compute the threshold
            parameter of the pruning based on the weights statistics.
        t_max: float
            Real (positive or negative) parameter used to compute the threshold
            parameter of the pruning based on the weights statistics.
    """
    # Defining the ReLU and Sigmoid functions
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Defining the thresholds
    x_mean, x_std = x.mean(), x.std()
    delta_min = (x_mean + t_min*x_std).abs()
    delta_max = (x_mean + t_max*x_std).abs()

    # Computing the output
    res = relu(x-delta_max)+delta_max*sigmoid(alpha*(x-delta_max)) - relu(-x-delta_min)-delta_min*sigmoid(alpha*(-x-delta_min))

    return res, delta_min, delta_max

def pruning_function_pTTQ_GSIA_old(x, alpha, t_min, t_max, current_epoch, total_epochs):
    """
    Enhanced pruning function with Gradual Sparsity Increase and Annealing (GSIA).
    
    Arguments:
    ----------
    x: torch.tensor
        Tensor to 'prune'
    alpha: float
        Hyper-parameter defining the 'speed' of the pruning.
    t_min: float
        Real parameter used to compute the lower threshold.
    t_max: float
        Real parameter used to compute the upper threshold.
    current_epoch: int
        Current training epoch.
    total_epochs: int
        Total number of training epochs.
    """

    # Defining the ReLU and Sigmoid functions
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Gradual sparsity increase factor
    #sparsity_factor = min(1.0, current_epoch / (total_epochs))
    sparsity_factor = 1.0 - torch.exp(-0.1 * current_epoch / total_epochs)

    # Annealing schedule for pruning threshold
    #annealing_factor = 1 - (current_epoch / total_epochs)**2
    annealing_factor = 1.0 - 0.2 * (current_epoch / total_epochs)**3
    if(sparsity_factor > 1):
        print(f"-- | Current epoch : {current_epoch}/ {total_epochs} | Annealing factor : {annealing_factor} | Sparsity factor : {sparsity_factor} | --")
    # Defining the thresholds
    x_mean, x_std = x.mean(), x.std()
    delta_min = (x_mean + t_min * x_std).abs() * annealing_factor
    delta_max = (x_mean + t_max * x_std).abs() * annealing_factor

    # Computing the output with gradual sparsity increase
    res = relu(x - delta_max) + delta_max * sigmoid(alpha * sparsity_factor * (x - delta_max)) - \
          relu(-x - delta_min) - delta_min * sigmoid(alpha * sparsity_factor * (-x - delta_min))

    return res


import json
import os
import hashlib


# EMA V3
def pruning_function_pTTQ_experimental_track(x, alpha, t_min, t_max,k=1):
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Compute statistics
    x_mean, x_std = x.mean(), x.std()
    
    # Compute adaptive thresholds with exponential moving average
    with torch.no_grad():
        if not hasattr(pruning_function_pTTQ_experimental, 'ema_min'):
            pruning_function_pTTQ_experimental.ema_min = (x_mean + t_min * x_std).abs()
            pruning_function_pTTQ_experimental.ema_max = (x_mean + t_max * x_std).abs()
        else:
            beta = 0.9  # EMA decay factor
            pruning_function_pTTQ_experimental.ema_min = beta * pruning_function_pTTQ_experimental.ema_min + (1 - beta) * (x_mean + t_min * x_std).abs()
            pruning_function_pTTQ_experimental.ema_max = beta * pruning_function_pTTQ_experimental.ema_max + (1 - beta) * (x_mean + t_max * x_std).abs()

    delta_min = pruning_function_pTTQ_experimental.ema_min
    delta_max = pruning_function_pTTQ_experimental.ema_max

    # Introduce a tunable constant to temper pruning aggressiveness
    #k = 1  # This value can be adjusted between 0 and 1

    # Apply pruning with adaptive thresholds and tempered aggressiveness
    res = relu(x - k * delta_max) + k * delta_max * sigmoid(alpha * (x - k * delta_max)) - \
          relu(-x - k * delta_min) - k * delta_min * sigmoid(alpha * (-x - k * delta_min))

    return res, delta_min, delta_max

def pruning_function_pTTQ_experimental(x, alpha, t_min, t_max,k=1, beta = 0.9):
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Compute statistics
    x_mean, x_std = x.mean(), x.std()
    
    # Compute adaptive thresholds with exponential moving average
    with torch.no_grad():
        if not hasattr(pruning_function_pTTQ_experimental, 'ema_min'):
            pruning_function_pTTQ_experimental.ema_min = (x_mean + t_min * x_std).abs()
            pruning_function_pTTQ_experimental.ema_max = (x_mean + t_max * x_std).abs()
        else:
            pruning_function_pTTQ_experimental.ema_min = beta * pruning_function_pTTQ_experimental.ema_min + (1 - beta) * (x_mean + t_min * x_std).abs()
            pruning_function_pTTQ_experimental.ema_max = beta * pruning_function_pTTQ_experimental.ema_max + (1 - beta) * (x_mean + t_max * x_std).abs()

    delta_min = pruning_function_pTTQ_experimental.ema_min
    delta_max = pruning_function_pTTQ_experimental.ema_max

    # Introduce a tunable constant to temper pruning aggressiveness
    #k = 1  # This value can be adjusted between 0 and 1

    # Apply pruning with adaptive thresholds and tempered aggressiveness
    res = relu(x - k * delta_max) + k * delta_max * sigmoid(alpha * (x - k * delta_max)) - \
          relu(-x - k * delta_min) - k * delta_min * sigmoid(alpha * (-x - k * delta_min))

    return res



def pruning_function_pTTQ_experimental_learned(x, alpha, a, b, k=1):
    """
        Pruning function similar to pruning_function_pTTQ_experimental,
        but thresholds are learnable instead of being computed via EMA.

        Arguments:
        ----------
        x: torch.tensor
            Tensor to 'prune'.
        alpha: float
            Hyper-parameter defining the 'speed' of the pruning.
        a: float
            Learnable NON-NEGATIVE threshold parameter for the lower bound.
        b: float
            Learnable NON-NEGATIVE threshold parameter for the upper bound.
        k: float
            Tunable constant to temper pruning aggressiveness (default is 1).
    """
    # Verifying that the values of a and b are non-negative
    if (type(a) != torch.Tensor) and (type(b) != torch.Tensor):
        assert (a >= 0) and (b >= 0)  # Cannot be used for tensors

    # Define the ReLU and Sigmoid functions
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Compute the output using learnable thresholds
    res = relu(x - k * b) + k * b * sigmoid(alpha * (x - k * b)) - \
          relu(-x - k * a) - k * a * sigmoid(alpha * (-x - k * a))

    return res

# EMA V2
def pruning_function_pTTQ_ema_v2(x, alpha, t_min, t_max):
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Compute statistics
    x_mean, x_std = x.mean(), x.std()
    
    # Compute adaptive thresholds with a smoothing factor
    delta_min = (x_mean + t_min * x_std).abs()
    delta_max = (x_mean + t_max * x_std).abs()
    
    # Introduce a smoothing factor based on the weight distribution
    smoothing_factor = torch.tanh(x_std)
    
    # Compute adaptive alpha
    alpha_adaptive = alpha * (1 + 0.1 * torch.tanh(x_std - 1))

    # Apply pruning with smoothed thresholds and adaptive alpha
    res = relu(x - smoothing_factor * delta_max) + smoothing_factor * delta_max * sigmoid(alpha_adaptive * (x - smoothing_factor * delta_max)) - \
          relu(-x - smoothing_factor * delta_min) - smoothing_factor * delta_min * sigmoid(alpha_adaptive * (-x - smoothing_factor * delta_min))

    return res

# EMA v1
def pruning_function_pTTQ_v1(x, alpha, t_min, t_max):
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Compute statistics
    x_mean, x_std = x.mean(), x.std()
    
    # Compute adaptive thresholds with exponential moving average
    with torch.no_grad():
        if not hasattr(pruning_function_pTTQ_experimental, 'ema_min'):
            pruning_function_pTTQ_experimental.ema_min = (x_mean + t_min * x_std).abs()
            pruning_function_pTTQ_experimental.ema_max = (x_mean + t_max * x_std).abs()
        else:
            beta = 0.9  # EMA decay factor
            pruning_function_pTTQ_experimental.ema_min = beta * pruning_function_pTTQ_experimental.ema_min + (1 - beta) * (x_mean + t_min * x_std).abs()
            pruning_function_pTTQ_experimental.ema_max = beta * pruning_function_pTTQ_experimental.ema_max + (1 - beta) * (x_mean + t_max * x_std).abs()

    delta_min = pruning_function_pTTQ_experimental.ema_min
    delta_max = pruning_function_pTTQ_experimental.ema_max

    # Compute adaptive alpha
    alpha_adaptive = alpha * (1 + torch.tanh(x_std - 1))

    # Apply pruning with adaptive thresholds and alpha
    res = relu(x - delta_max) + delta_max * sigmoid(alpha_adaptive * (x - delta_max)) - \
          relu(-x - delta_min) - delta_min * sigmoid(alpha_adaptive * (-x - delta_min))

    return res

def pruning_function_pTTQ_adaptive_v2(x, alpha, t_min, t_max):
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Compute statistics
    x_mean, x_std = x.mean(), x.std()
    
    # Compute adaptive thresholds
    q1, q3 = torch.quantile(x, torch.tensor([0.25, 0.75]))
    iqr = q3 - q1
    delta_min = torch.max(x_mean + t_min * x_std, q1 - 1.5 * iqr).abs()
    delta_max = torch.min(x_mean + t_max * x_std, q3 + 1.5 * iqr).abs()

    # Compute smoothing factor
    beta = torch.clamp(1 - (delta_max - delta_min) / (2 * x_std), 0.5, 1.0)

    # Apply pruning with smoothed thresholds
    res = relu(x - beta * delta_max) + beta * delta_max * sigmoid(alpha * (x - beta * delta_max)) - \
          relu(-x - beta * delta_min) - beta * delta_min * sigmoid(alpha * (-x - beta * delta_min))

    return res

def pruning_function_pTTQ_GSIA(x, alpha, t_min, t_max, layer_id):
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    x_mean, x_std = x.mean(), x.std()
    delta_min = (x_mean + t_min*x_std).abs()
    delta_max = (x_mean + t_max*x_std).abs()

    # Layer-adaptive threshold adjustment
    layer_factor = 1 + 0.1 * torch.tanh(torch.tensor(layer_id / 10))
    delta_min *= layer_factor
    delta_max *= layer_factor

    # Compute pruned weights
    pruned = relu(x-delta_max) + delta_max*sigmoid(alpha*(x-delta_max)) - \
             relu(-x-delta_min) - delta_min*sigmoid(alpha*(-x-delta_min))

    # Add regularization term
    reg_term = 0.01 * torch.sum(torch.abs(pruned)) / torch.sum(torch.abs(x))
    
    return pruned, reg_term
"""
def pruning_function_pTTQ_GSIA(x, alpha, t_min, t_max, current_epoch, total_epochs):
    
    Enhanced pruning function with Adaptive Sparsity and Gentle Annealing (ASGA).
    
    Arguments:
    ----------
    x: torch.tensor
        Tensor to 'prune'
    alpha: float
        Hyper-parameter defining the 'speed' of the pruning.
    t_min: float
        Real parameter used to compute the lower threshold.
    t_max: float
        Real parameter used to compute the upper threshold.
    current_epoch: int
        Current training epoch.
    total_epochs: int
        Total number of training epochs.
    
    # Defining the ReLU and Sigmoid functions
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Convert scalars to tensors
    current_epoch_tensor = torch.tensor(current_epoch, dtype=torch.float32, device=x.device)
    total_epochs_tensor = torch.tensor(total_epochs, dtype=torch.float32, device=x.device)

    # Adaptive sparsity factor
    sparsity_factor = 1.0 - torch.exp(-0.1 * current_epoch_tensor / total_epochs_tensor)
    
    # Gentle annealing schedule
    annealing_factor = 1.0 - 0.2 * (current_epoch_tensor / total_epochs_tensor)**3

    # Defining the thresholds
    x_mean, x_std = x.mean(), x.std()
    delta_min = (x_mean + t_min * x_std).abs() * annealing_factor
    delta_max = (x_mean + t_max * x_std).abs() * annealing_factor

    # Computing the output with adaptive sparsity
    res = relu(x - delta_max) + delta_max * sigmoid(alpha * sparsity_factor * (x - delta_max)) - \
          relu(-x - delta_min) - delta_min * sigmoid(alpha * sparsity_factor * (-x - delta_min))
"""
def pruning_function_asymmetric_manessi(x, alpha, a, b):
    """
        Function inspired from the work of Manessi et al. (2019)
        Compute a pruning function of the input tensor x
        based on two threshold a and b, at a "speed" alpha.
        WARNING: there is not actual pruning that is done, but
        the value of x is set very close to zero if it is
        in an interval defined by a and b.

        Arguments:
        ----------
        x: torch.tensor
            Tensor to 'prune'
        alpha: float
            Hyper-parameter defining the 'speed' of the pruning.
        a: float
            NON-NEGATIVE threshold parameter of the 'pruning'
        b: float
            NON-NEGATIVE threshold parameter of the 'pruning'
    """
    # Verifying that the values of a and b are non-negative
    if (type(a) != torch.Tensor) and (type(b) != torch.Tensor):
        assert (a >= 0) and (b >= 0) # Cannot be used for tensors
    # Defining the ReLU and Sigmoid functions
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()

    # Computing the output
    res = relu(x-b)+b*sigmoid(alpha*(x-b)) - relu(-x-a)-a*sigmoid(alpha*(-x-a))

    return res
