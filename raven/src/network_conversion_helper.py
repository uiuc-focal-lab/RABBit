from enum import Enum

# import the necessary packages
import torch.nn
from torch.nn import ModuleList
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Flatten
from torch.nn import LogSoftmax
from torch.nn import Sequential
from torch import flatten
from onnx import numpy_helper

from raven.src.networks import FullyConnected, Conv
from raven.src.common.network import LayerType


class TransformedNet(Module):
    def __init__(self, layers, ignore_last_layer=False, all_linear=False):
        super().__init__()
        constructed_layers = []
        self.conv_layer_count = 0
        self.linear_layer_count = 0
        # Initially the there will be only one channel.
        self.last_conv_layer_channels = 1
        if all_linear:
            constructed_layers.append(Flatten(start_dim=1))
        for layer in layers:
            if layer.type == LayerType.Linear:
                if self.linear_layer_count == 0 and self.conv_layer_count > 0:
                    constructed_layers.append(Flatten(start_dim=1))
                self.linear_layer_count += 1
                shape = layer.weight.shape
                if shape is None:
                    raise ValueError("Shape of the linear layer should be not null")
                input_len = shape[1]
                output_len = shape[0]
                constructed_layers.append(Linear(input_len, output_len))
                constructed_layers[-1].weight = torch.nn.Parameter(layer.weight)
                constructed_layers[-1].bias = torch.nn.Parameter(layer.bias)
            elif layer.type == LayerType.Conv2D:
                self.conv_layer_count += 1
                kernel_size = layer.kernel_size
                padding = layer.padding
                stride = layer.stride
                dilation = layer.dilation
                in_channels = self.last_conv_layer_channels
                out_channels = layer.weight.shape[0]
                self.last_conv_layer_channels = out_channels
                constructed_layers.append(Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, dilation=dilation))
                constructed_layers[-1].weight = torch.nn.Parameter(layer.weight)
                constructed_layers[-1].bias = torch.nn.Parameter(layer.bias)
            elif layer.type == LayerType.ReLU:
                constructed_layers.append(ReLU())
            else:
                raise NotImplementedError("Layer conversion of type {} is not supported".format(layer.type))
        if ignore_last_layer:
            _ = constructed_layers.pop()
        self.model = Sequential(*constructed_layers)

    def forward(self, x):
        return self.model(x)


def convert_model(parsed_net, remove_last_layer=True, all_linear=False):
    return TransformedNet(parsed_net, remove_last_layer, all_linear)


def is_linear(net_name):
    if net_name == 'mnist-net_256x2.onnx':
        return True
    else:
        return False

def get_pytorch_net(model, remove_last_layer, all_linear):
    converted_model = convert_model(parsed_net=model, remove_last_layer=remove_last_layer, all_linear=all_linear)
    return converted_model
