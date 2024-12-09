import onnx
from onnx import numpy_helper
import numpy as np
import torch
import src.parse as parse
from src.common.network import Layer, LayerType, Network

class PruneArgs:
    def __init__(self, epsilon=0.1, net=None, layer_index=-1, layer_name=None, unstructured_prune=True, passive=True, passive_relus=[], net_type='custom'):
        self.epsilon = epsilon 
        self.layer_index = layer_index 
        self.layer_name = layer_name 
        self.net = net 
        self.net_type = net_type 
        self.unstructured_prune = unstructured_prune
        self.passive = passive 
        self.passive_relus = passive_relus

    def __repr__(self):
        return "epsilon: "+str(self.epsilon) + "\nlayer_index: "+str(self.layer_index) + "\nnet_type: "+str(self.net_type) + "\nunstructured_prune: "+str(self.unstructured_prune) + "\npassive: "+str(self.passive) + "\npassive_relus: "+str(self.passive_relus)
    

def get_layer_onnx_to_numpy(net, layer_index=None, name=None):
    initializers = net.graph.initializer
    layer = None
    if name is not None:
        for i, initializer in enumerate(initializers):
            if initializer.name == name:
                layer = initializer
                layer_index = i
                break 
    elif layer_index is not None:
        layer = initializers[layer_index]
        name = layer.name 
    layer = numpy_helper.to_array(layer)
    layer = np.array(layer)
    return layer, layer_index, name 

def replace_onnx_layer(net, new_layer, layer_index, name):
    new_layer = numpy_helper.from_array(new_layer, name=name)
    net.graph.initializer[layer_index].CopyFrom(new_layer)
    # return net  

def print_pruning_stats(layer, index, cumsum, epsilon, layer_index):
    print('No. of weights pruned: ', index+1)
    print('Total weights: ', layer.shape[0] * layer.shape[1])
    print('Weights in the layer', layer_index)
    print('Precentage of weights pruned: ', (index+1)*100/(layer.shape[0] * layer.shape[1]))
    print('deviation allowed: ', epsilon)
    print('actual deviation: ', np.sqrt(cumsum))

def prune_layer_weights(layer, epsilon, layer_index):
    layer_squared = np.square(layer)
    weights_flattened = list()
    for row in range(layer_squared.shape[0]):
        for column in range(layer_squared[row].shape[0]):
            weights_flattened.append((layer_squared[row][column], row, column))
    weights_flattened.sort()
    index = -1
    cumsum = 0
    for i in range(len(weights_flattened)):
        if cumsum + weights_flattened[i][0] >= epsilon**2:
            break 
        cumsum += weights_flattened[i][0]
        index += 1

    i = 0
    while(i <= index):
        row = weights_flattened[i][1]
        column = weights_flattened[i][2]
        layer[row,column] = 0
        i += 1
    print_pruning_stats(layer, index, cumsum, epsilon, layer_index)
    return layer, index, cumsum

def prune_passive_relus(in_index, out_index, net, relus):
    in_layer = net[in_index]
    out_layer = net[out_index]
    if in_layer.type is not LayerType.Linear or out_layer.type is not LayerType.Linear:
        return 

    for i in relus:
        in_layer.bias[i] = 0
        w = in_layer.weight[i]
        w[w!=0] = 0
        out_layer.weight[:, i] = 0


def prune_last_layer(weight, indices):
    sz = weight.size()
    for ind in indices:
        if ind < sz[1]:
            with torch.no_grad():
                weight[:, ind] = 0
        else:
            raise ValueError("Inidices out of range")


# For custom net types returns the list of linear layers of the
# Network.
def get_linear_layers(net):
    linear_layers = []
    for layer in net:
        if layer.type is not LayerType.Linear:
            continue
        linear_layers.append(layer)
    return linear_layers

def prune_redundant_neurons(net):
    linear_layers = get_linear_layers(net)
    count = 0
    for i in range(1, len(linear_layers)):
        weight = linear_layers[i].weight
        size = weight.size()
        for j in range(size[1]):
            if torch.count_nonzero(weight[:, j]).item() == 0:
                if i == len(linear_layers) -1:
                    count += 1
                linear_layers[i-1].weight[j] = 0.0
                linear_layers[i-1].bias[j] = 0.0
    # print("No of redundant neurons found", count)



def print_stats(net, stat_dictionary):
    layer_indx = 1
    for layer in net:
        if layer.type is not LayerType.Linear:
            continue
        total_weights = torch.numel(layer.weight)
        non_zero_weights = torch.count_nonzero(layer.weight)
        pertentage_pruned = (1.0 - (non_zero_weights / total_weights)) * 100.0
        print("Layer index {} pruned weight {} ".format(layer_indx, pertentage_pruned))
        if layer_indx not in stat_dictionary.keys():
            stat_dictionary[layer_indx] = []
        stat_dictionary[layer_indx].append(pertentage_pruned.item())
        layer_indx += 1     

def get_pruned_network(prune_args):
    # READ ARGS
    net = prune_args.net 
    layer_index = prune_args.layer_index 
    layer_name = prune_args.layer_name 
    epsilon = prune_args.epsilon 

    if prune_args.unstructured_prune:
        # CUSTOM NETWORK TYPE
        if epsilon is None:
            return net
        if prune_args.net_type == 'custom':
            linear_layers = get_linear_layers(net)
            layer = linear_layers[layer_index].weight.numpy() 
            layer, index, cumsum = prune_layer_weights(layer, epsilon, layer_index)
            net[layer_index].weight = torch.from_numpy(layer) 
            return net 

        # ONNX NETWORK TYPE
        elif prune_args.net_type == 'onnx':
            layer, layer_index, name = get_layer_onnx_to_numpy(net, layer_index, layer_name)
            layer, index, cumsum = prune_layer_weights(layer, epsilon, layer_index)
            replace_onnx_layer(net, layer, layer_index, name)
            return net
        
        else:
            raise Exception("network type unknown", prune_args.net_type)
    else:
        if prune_args.passive:
            if prune_args.net_type == 'custom':
                passive_relus = prune_args.passive_relus 
                index_ctr = 0
                for i in range(len(passive_relus)):
                    while net[index_ctr].type!=LayerType.ReLU:
                        index_ctr += 1
                    layer_index = index_ctr 
                    index_ctr += 1
                    in_index = layer_index - 1
                    out_index = layer_index + 1
                    prune_passive_relus(in_index, out_index, net, passive_relus[i])
                    prune_redundant_neurons(net)
                return net
            else:
                raise Exception("network type not implemented", prune_args.net_type)