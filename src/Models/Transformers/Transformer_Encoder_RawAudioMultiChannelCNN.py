#!/usr/bin/env python3

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchinfo import summary

################################################################################
################################################################################

class PositionalEncoding(nn.Module):
    """
        Modified version of the code in https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        to put the batch first
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, pos_encoder_type='Original'):
        super().__init__()
        self.pos_encoder_type = pos_encoder_type
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if (d_model % 2 == 1):
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        try:
            if (self.pos_encoder_type.lower() == 'mlhc'):
                x = x + self.pe[:, x.size(1), :]
            elif (self.pos_encoder_type.lower() == 'original'):
                x = x + self.pe[:, 0:x.size(1), :]
        except:
            # Using the one from ORIGINAL
            print("\n\nWARNING: USING ORIGINAL POSITIONAL ENCODING BECAUSE NO POSITIONAL ENCODING WAS FOUND\n\n")
            x = x + self.pe[:, 0:x.size(1), :]

        return self.dropout(x)

class TransformerEncoderMultichannelCNN(nn.Module):

    def __init__(
                    self,
                    in_channels: int,
                    nhead: int,
                    d_hid: int,
                    nlayers: int,
                    dropout: float = 0.5,
                    d_model: int = 64,
                    classification_pool: str = 'ClassToken',
                    n_conv_layers: int = 2,
                    pos_encoder_type: str = 'Original'
                ):
        super().__init__()
        # Parameters of the model
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.classification_pool = classification_pool
        self.in_channels = in_channels
        self.pos_encoder_type = pos_encoder_type

        # Layers
        # Input embedding layers
        self.n_conv_layers = n_conv_layers
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=self.d_model, kernel_size=3, stride=1, padding=1)
        self.conv = torch.nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, stride=1, padding=0)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)

        # Class token parameter (if used)
        if (self.classification_pool.lower() == 'classtoken'):
            self.class_token = nn.Parameter(torch.randn(self.d_model)) # Normal random tensor

        # Positional Encoder Layer
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, pos_encoder_type=self.pos_encoder_type)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
        """
        # Extraction of the embeddings
        x = F.leaky_relu(self.conv1(src))
        x = F.leaky_relu(self.conv2(x))
        for i in range(self.n_conv_layers):
          x = F.leaky_relu(self.conv(x))
          x = self.maxpool(x)
        x = torch.swapaxes(x, 1, 2) # Becuase the input of the PE and the Transformer
        # has to be under the format (batch_size, seq_len, feat_dim)

        # Class token if used
        if (self.classification_pool.lower() == 'classtoken'):
            batch_size = x.shape[0]
            class_tokens = self.class_token.expand(batch_size, -1)
            class_tokens = class_tokens.unsqueeze(dim=1)
            x = torch.cat([class_tokens, x], dim=1)


        # Positional encoding
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer encoding
        encoding = self.transformer_encoder(x)

        # Embedded representation
        if (self.classification_pool.lower() == 'classtoken'):
            class_hidden_features = encoding[:, 0]
        else:
            class_hidden_features = encoding

        return class_hidden_features


class TransformerClassifierMultichannelCNN(nn.Module):

    def __init__(
                    self,
                    in_channels: int,
                    nhead: int,
                    d_hid: int,
                    nlayers: int,
                    dropout: float = 0.5,
                    nb_features_projection: float = 50,
                    d_model: int = 64,
                    num_classes: int = 3,
                    classification_pool: str = 'ClassToken',
                    n_conv_layers: int = 2,
                    pos_encoder_type: str = 'Original'
                ):
        super().__init__()
        # Parameters of the model
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.classification_pool = classification_pool
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pos_encoder_type = pos_encoder_type

        # Encoder model
        self.encoder = TransformerEncoderMultichannelCNN(
                                                            in_channels,
                                                            nhead,
                                                            d_hid,
                                                            nlayers,
                                                            dropout,
                                                            d_model,
                                                            classification_pool,
                                                            n_conv_layers,
                                                            pos_encoder_type
                                                        )
        self.d_model = self.encoder.d_model

        # Classification layers
        self.layer_norm_1 = nn.LayerNorm(self.d_model)
        self.out_1 = nn.Linear(self.d_model, nb_features_projection)
        self.layer_norm_2 = nn.LayerNorm(nb_features_projection)
        self.out_2 = nn.Linear(nb_features_projection, self.num_classes)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
        """
        # Encoding using the Transformer model
        class_hidden_features = self.encoder(src)

        # Classification layers
        output = self.out_1(self.layer_norm_1(class_hidden_features))
        output = self.out_2(self.layer_norm_2(output))
        if (self.classification_pool.lower() == 'avg'):
            output = output.mean(dim=1)

        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


################################################################################
################################################################################
"""
    MAIN CLASS
"""
if __name__=='__main__':
    # Device to use for computation
    device = torch.device("cpu")

    # Creating the model
    in_channels = 1 # ESR
    nhead = 4 # ESR
    d_hid = 8 # ESR
    nlayers = 4 # ESR
    dropout = 0.3 # ESR
    nb_features_projection = 4 # ESR
    d_model = 64 # ESR
    # classification_pool = 'Avg'
    classification_pool = 'ClassToken'
    n_conv_layers = 2 # ESR
    num_classes = 2 # ESR
    pos_encoder_type = 'Original'
    # pos_encoder_type = 'MLHC'
    model = TransformerClassifierMultichannelCNN(
                                                    in_channels,
                                                    nhead,
                                                    d_hid,
                                                    nlayers,
                                                    dropout,
                                                    nb_features_projection,
                                                    d_model,
                                                    num_classes,
                                                    classification_pool,
                                                    n_conv_layers,
                                                    pos_encoder_type
                                                )
    model.to(device)

    # # Creating dummy data
    audio_lenght = 178 # ESR
    bs = 1
    dummy_data = torch.randn((bs, in_channels, audio_lenght)).to(device)
    print("Shape of the input data: {}".format(dummy_data.shape))

    # Testing the model with dummy data
    output = model(dummy_data)
    print("Output shape: ", output.shape)

    # Summary of the model
    summary(model, (bs, in_channels, audio_lenght))
