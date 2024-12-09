from enum import Enum


class RavenMode(Enum):
    UAP = 1
    UAP_TARGETED = 2
    ENSEMBLE = 3
    UAP_BINARY = 4


class Dataset(Enum):
    MNIST = 1
    CIFAR10 = 2
