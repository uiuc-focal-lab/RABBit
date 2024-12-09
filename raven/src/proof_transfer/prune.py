import onnx
from onnx import numpy_helper
import numpy as np

PATH = 'ACASXU_run2a_1_1_batch_2000.onnx'
epsilon = 0.0156
layer_id = -2
layer_name = 'W6'

def save_net(net, name):
    onnx.save(net, name)

def replace_layer(net, new_layer, layer_id, name):
    new_layer = numpy_helper.from_array(new_layer, name=name)
    net.graph.initializer[layer_id].CopyFrom(new_layer)
    return net     

def get_layer_numpy(net, layer_id=None, name=None):
    initializers = net.graph.initializer
    layer = None
    if name is not None:
        for i, initializer in enumerate(initializers):
            if initializer.name == name:
                layer = initializer
                layer_id = i
                break 
    elif layer_id is not None:
        layer = initializers[layer_id]
        name = layer.name 
    layer = numpy_helper.to_array(layer)
    layer = np.array(layer)
    return layer, layer_id, name 

def print_pruning_stats(layer, index, cumsum, epsilon):
    print('No. of weights pruned: ', index+1)
    print('Total weights: ', layer.shape[0] * layer.shape[1])
    print('Precentage of weights pruned: ', (index+1)*100/(layer.shape[0] * layer.shape[1]))
    print('deviation allowed: ', epsilon)
    print('actual deviation: ', np.sqrt(cumsum))

def prune_layer_weights(layer, epsilon):
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
    print_pruning_stats(layer, index, cumsum, epsilon)
    return layer, index, cumsum


def prune_unstructured(net, epsilon, layer_id=None, name=None):
    layer, layer_id, name = get_layer_numpy(net, layer_id, name)
    layer, index, cumsum = prune_layer_weights(layer, epsilon)
    return replace_layer(net, layer, layer_id, name)

# net = onnx.load(PATH)
# net = prune_unstructured(net, epsilon, name=layer_name)
# save_net(net, 'new.onnx')