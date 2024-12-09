import torch
from src.common import Dataset
import raven.src.util as util
import raven.src.config as config


def get_net(net_names, dataset : Dataset):
    nets = []
    for net_name in net_names:
        net_path = config.NET_HOME + net_name
        nets.append(util.get_net(net_path, dataset))
    return nets