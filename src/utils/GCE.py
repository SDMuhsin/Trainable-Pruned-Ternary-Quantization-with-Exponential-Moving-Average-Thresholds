#!/usr/bin/env python3

import torch
import torch.nn as nn

class GeneralizedCrossEntropy(torch.nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7, class_weights = None) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-9
        self.softmax = torch.nn.Softmax(dim=1)
        self.class_weights = class_weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q

        # Weighting the classes if class weights are given
        if (self.class_weights is not None):
            device_to_use = loss.device
            class_weight_vector = torch.tensor([self.class_weights[true_class] for true_class in target]).to(device_to_use)
            loss = class_weight_vector*loss

        return torch.mean(loss)
