from raven.src.specs.input_spec import InputSpecType
from enum import IntEnum
from src.common import Dataset

NET_HOME = "nets/"
DEVICE = 'cpu'
GPU_DEVICE_LIST = []


class Args:
    def __init__(self, net, domain, count=None, eps=0.01, dataset='mnist', spec_type=InputSpecType.LINF, split=None,
                 pt_method=None, timeout=None, parallel=False, initial_split=False, attack=None):
        self.net = NET_HOME + net
        self.domain = domain
        self.count = count
        self.eps = eps
        self.dataset = dataset
        self.spec_type = spec_type
        self.split = split
        self.pt_method = pt_method
        self.timeout = timeout
        self.parallel = parallel
        self.initial_split = initial_split
        self.attack = attack
        self.ignore_properties = []


class PruningArgs:
    def __init__(self, desried_perturbation=None, layers_to_prune=None, swap_layers= False, node_wise_bounds=False,
                        unstructured_pruning=True, structured_pruning=False, maximum_iteration=10, accuracy_drop=None):
        self.desired_perturbation= desried_perturbation
        self.layers_to_prune = layers_to_prune
        self.swap_layers = swap_layers
        self.node_wise_bounds= node_wise_bounds
        self.unstructured_pruning = unstructured_pruning
        self.structured_pruning = structured_pruning
        self.maximum_iteration = maximum_iteration
        self.accuracy_drop = accuracy_drop

log_file = "log.txt"
log_enabled = False


def write_log(log):
    """Appends string @param: str to log file"""
    if log_enabled:
        f = open(log_file, "a")
        f.write(log + '\n')
        f.close()


tool_name = "IVAN"
baseline = "Baseline"

linear_models = []

def is_linear_model(net_name):
    for name in linear_models:
        if name in net_name:
            return True
    return False

MNIST_NO_TRANSFORM_NETS = []


def mnist_data_transform(dataset, net_name):
    if dataset == Dataset.MNIST:
        for name in MNIST_NO_TRANSFORM_NETS:
            if name in net_name:
                return False
        return True
    elif dataset == Dataset.CIFAR10:
        return True
    else:
        raise ValueError(f'Unsupported dataset {dataset}')


class MONOTONE_PROP(IntEnum):
    CRIM = 0
    ZN = 1
    INDUS = 2
    NOX = 3
    RM = 4
    AGE = 5
    DIS = 6
    RAD = 7
    TAX = 8
    PTRATIO = 9
    B = 10
    LSTAT = 11


def ACASXU(i, j):
    net_name = "acasxu/nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    return net_name
