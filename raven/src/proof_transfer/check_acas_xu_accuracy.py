import sys
import torch

from src.proof_transfer.prune_network import get_linear_layers

sys.path.append('nn-proof-transfer/nn_pruning/src')
sys.path.append('nn-proof-transfer/acasxu-art/ART/art')

from exp_acas import eval_test, AcasPoints
from acas import AcasNetID, AcasNet
from diffabs import DeeppolyDom

def check_acas_accuracy(pt_args, passed_net):
    x, y =pt_args.get_acas_xu_indices()
    nid = AcasNetID(x, y)
    linear_layers = get_linear_layers(passed_net)
    layer_weights = []
    layer_biases = []
    for layer in linear_layers:
        layer_weights.append(layer.weight)
        layer_biases.append(layer.bias)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dom = DeeppolyDom()
    fpath = nid.fpath()
    net, _, _ = AcasNet.load_nnet(fpath, dom, device)
    net.update_weight_and_bias(layer_weights, layer_biases)
    testset = AcasPoints.load(nid, train=False, device=device)
    accuracy = eval_test(net, testset)
    print("Accuracy", accuracy)
    return accuracy


# if __name__ == '__main__':
#     pt_args = TransferArgs(domain=None, approx=None, dataset=None)
#     pt_args.set_acas_xu_indices(1, 1)
#     check_acas_accuracy(pt_args=pt_args, passed_net=None)

