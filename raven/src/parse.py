import torch
import raven.src.common as common
import raven.src.util as util

from torch.nn import ReLU, Linear, Conv2d, Flatten

from raven.src.domains.lptransformer import LPTransformer
from onnx import numpy_helper
from raven.src.common.network import Layer, LayerType, Network


def get_transformer(transformer, net, prop, relu_mask=None):
    if type(transformer) is LPTransformer:
        if net[-1].type != LayerType.Linear:
            raise ValueError("We assume that the last layer of the network is a Linear layer!")
        transformer.handle_network(net, prop, relu_mask=relu_mask)
        return transformer

    # For all abstraction based domains (all other domains than lp)
    transformer = forward_layers(net, relu_mask, transformer)

    return transformer


def forward_layers(net, relu_mask, transformers):
    for layer in net:
        if layer.type == LayerType.ReLU:
            transformers.handle_relu(layer, optimize=True, relu_mask=relu_mask)
        elif layer.type == LayerType.Linear:
            if layer == net[-1]:
                transformers.handle_linear(layer, last_layer=True)
            else:
                transformers.handle_linear(layer)
        elif layer.type == LayerType.Conv2D:
            transformers.handle_conv2d(layer)
        elif layer.type == LayerType.Normalization:
            transformers.handle_normalization(layer)
    return transformers


def parse_onnx_layers(net):
    input_shape = tuple([dim.dim_value for dim in net.graph.input[0].type.tensor_type.shape.dim])
    # Create the new Network object
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape)
    num_layers = len(net.graph.node)
    model_name_to_val_dict = {init_vals.name: torch.tensor(numpy_helper.to_array(init_vals)) for init_vals in
                              net.graph.initializer}

    final_relu = False
    for cur_layer in range(num_layers):
        final_relu = False
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input
        if operation == 'MatMul':
            # Assuming that the add node is followed by the MatMul node
            add_node = net.graph.node[cur_layer + 1]
            bias = model_name_to_val_dict[add_node.input[1]]

            # Making some weird assumption that the weight is always 0th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[0]], bias=bias, type=LayerType.Linear)
            layers.append(layer)

        elif operation == 'Conv':
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Conv2D)
            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            layers.append(layer)

        elif operation == 'Gemm':
            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Linear)
            layers.append(layer)

        elif operation == 'Relu':
            final_relu = True
            layers.append(Layer(type=LayerType.ReLU))
    
    # The final most layer is relu and no linear layer after that
    # remove the linear layer (Hack find better solutions).
    if final_relu is True:
        layers.pop()
    return layers


def parse_torch_layers(net):
    layers = Network(torch_net=net)

    for torch_layer in net:
        if isinstance(torch_layer, ReLU):
            layers.append(Layer(type=LayerType.ReLU))
        elif isinstance(torch_layer, Linear):
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias, type=LayerType.Linear)
            layers.append(layer)
        elif isinstance(torch_layer, Conv2d):
            layer = Layer(weight=torch_layer.weight, bias=torch_layer.bias,
                          type=LayerType.Conv2D)
            layer.kernel_size = torch_layer.kernel_size
            layer.padding = torch_layer.padding
            layer.stride = torch_layer.stride
            layer.dilation = (1, 1)
            layers.append(layer)

    return layers