from torch import nn


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: nn.Module
    ):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        return (
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn='relu',
                 batch_norm=False, dropout_rate=0.0):
        super(ConvLayer, self).__init__()

        # Convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # Optionally add Batch Normalization
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None

        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

        # Set activation function
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_fn == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)

        # Apply Batch Norm if available
        if self.batch_norm:
            x = self.batch_norm(x)

        # Apply activation function
        x = self.activation(x)

        # Apply Dropout if available
        if self.dropout:
            x = self.dropout(x)

        return x
