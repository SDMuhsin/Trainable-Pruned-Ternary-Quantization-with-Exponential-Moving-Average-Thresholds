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
def pruning_function_pTTQ_GSIA(x, alpha, t_min, t_max, current_epoch, total_epochs):
    """
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
    """
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
