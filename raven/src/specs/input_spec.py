import copy
import torch

from enum import Enum

from raven.src import parse
from raven.src.bnb import Split
from raven.src.domains.deeppoly import DeeppolyTransformer
from raven.src.domains.deepz import ZonoTransformer
from raven.src.specs.out_spec import OutSpecType
from raven.src.common.dataset import Dataset

class InputSpecType(Enum):
    LINF = 1
    PATCH = 2
    GLOBAL = 3
    UAP = 4
    UAP_TARGETED = 5

def get_mean_std(dataset):
    if dataset == Dataset.MNIST:
        means = [0]
        stds = [1]
    elif dataset == Dataset.CIFAR10:
        # For the model that is loaded from cert def this normalization was
        # used
        stds = [0.2023, 0.1994, 0.2010]
        means = [0.4914, 0.4822, 0.4465]
        # means = [0.0, 0.0, 0.0]
        # stds = [1, 1, 1]
    elif dataset == Dataset.ACAS:
        means = [19791.091, 0.0, 0.0, 650.0, 600.0]
        stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
    else:
        raise ValueError("Unsupported Dataset!")
    return torch.tensor(means).reshape(-1, 1, 1), torch.tensor(stds).reshape(-1, 1, 1)


class InputProperty(object):
    def __init__(self, input_lb, input_ub, out_constr, dataset, input=None):
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.out_constr = out_constr
        self.dataset = dataset
        if input is not None:
            self.input = input.flatten()
        else:
            self.input = None

    def update_input(self, eps):
        self.input += eps 
        self.input_lb += eps
        self.input_ub += eps

    
    def update_bounds(self, eps):
        ilb = torch.clip(self.input - eps, min=0., max=1.)
        iub = torch.clip(self.input + eps, min=0., max=1.)

        mean, std = get_mean_std(self.dataset)

        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        self.input = (self.input - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)
        self.input_lb = ilb
        self.input_ub = iub

    def __hash__(self):
        return hash((self.input_lb.numpy().tobytes(), self.input_ub.numpy().tobytes(), self.dataset))

    # After has collision Python dict check for equality. Thus, our definition of equality should define both
    def __eq__(self, other):
        if not torch.all(self.input_lb == other.input_lb) or not torch.all(self.input_ub == other.input_ub) \
                or self.dataset != other.dataset or not torch.all(self.out_constr.constr_mat[0] == other.out_constr.constr_mat[0]):
            return False
        return True

    def create_split_input_spec(self, input_lb, input_ub):
        return InputProperty(input_lb, input_ub, self.out_constr, self.dataset)

    def is_local_robustness(self):
        return self.out_constr.constr_type == OutSpecType.LOCAL_ROBUST

    def get_label(self):
        if self.out_constr.constr_type is not OutSpecType.LOCAL_ROBUST:
            raise ValueError("Label only for local robustness properties!")
        return self.out_constr.label

    def get_input_size(self):
        return self.input_lb.shape[0]

    def is_conjunctive(self):
        return self.out_constr.is_conjunctive

    def output_constr_mat(self):
        return self.out_constr.constr_mat[0]

    def output_constr_const(self):
        return self.out_constr.constr_mat[1]

    def split_spec(self, split, chosen_dim):
        if split == Split.INPUT or split == Split.INPUT_GRAD or split == Split.INPUT_SB:
            # Heuristic: Divide in 2 with longest width for now
            # choose a particular dimension of the input to split
            ilb1 = copy.deepcopy(self.input_lb)
            iub1 = copy.deepcopy(self.input_ub)

            iub1[chosen_dim] = (self.input_ub[chosen_dim] + self.input_lb[chosen_dim]) / 2

            ilb2 = copy.deepcopy(self.input_lb)
            iub2 = copy.deepcopy(self.input_ub)

            ilb2[chosen_dim] = (self.input_ub[chosen_dim] + self.input_lb[chosen_dim]) / 2

            return [self.create_split_input_spec(ilb1, iub1), self.create_split_input_spec(ilb2, iub2)]
        else:
            raise ValueError("Unsupported input split!")

    def multiple_splits(self, num_splits):
        all_splits = []
        new_ilb = copy.deepcopy(self.input_lb)
        new_iub = copy.deepcopy(self.input_ub)
        step_size = []
        # Assuming ACAS-XU for now
        for i in range(5):
            step_size.append((self.input_ub[i]-self.input_lb[i])/num_splits[i])

        for i in range(num_splits[0]):
            new_ilb[0] = self.input_lb[0] + i*step_size[0]
            new_iub[0] = self.input_lb[0] + (i+1)*step_size[0]
            for j in range(num_splits[1]):
                new_ilb[1] = self.input_lb[1] + j * step_size[1]
                new_iub[1] = self.input_lb[1] + (j + 1) * step_size[1]
                for k in range(num_splits[2]):
                    new_ilb[2] = self.input_lb[2] + k * step_size[2]
                    new_iub[2] = self.input_lb[2] + (k + 1) * step_size[2]
                    for l in range(num_splits[3]):
                        new_ilb[3] = self.input_lb[3] + l * step_size[3]
                        new_iub[3] = self.input_lb[3] + (l + 1) * step_size[3]
                        for m in range(num_splits[4]):
                            new_ilb[4] = self.input_lb[4] + m * step_size[4]
                            new_iub[4] = self.input_lb[4] + (m + 1) * step_size[4]

                            all_splits.append(self.create_split_input_spec(copy.deepcopy(new_ilb), copy.deepcopy(new_iub)))
        return all_splits

    def choose_split_dim(self, split, net=None):
        if split == Split.INPUT:
            chosen_dim = torch.argmax(self.input_ub - self.input_lb)
        elif split == Split.INPUT_GRAD:
            zono_transformer = ZonoTransformer(self, complete=True)
            zono_transformer = parse.get_transformer(zono_transformer, net, self)

            center = zono_transformer.centers[-1]
            cof = zono_transformer.cofs[-1]
            cof_abs = torch.sum(torch.abs(cof), dim=0)
            lb = center - cof_abs

            if self.out_constr.is_conjunctive:
                adv_index = torch.argmin(lb)
            else:
                adv_index = torch.argmax(lb)

            input_len = len(self.input_lb)
            chosen_noise_idx = torch.argmax(torch.abs(cof[:input_len, adv_index])).item()
            # chosen_noise_idx = torch.argmax(torch.sum(torch.abs(cof[:input_len]), dim=1) * (self.input_ub - self.input_lb))
            chosen_dim = zono_transformer.map_for_noise_indices[chosen_noise_idx]
        elif split == Split.INPUT_SB:
            cp_spec = copy.deepcopy(self)
            lb0 = self.get_zono_lb(net, cp_spec)

            chosen_dim = -1
            best_score = -1e-3

            for dim in range(len(self.input_lb)):
                s1, s2 = cp_spec.split_spec(split, dim)

                lb1 = self.get_zono_lb(net, s1)
                lb2 = self.get_zono_lb(net, s2)

                dim_score = min(lb1-lb0, lb2-lb0)

                if dim_score > best_score:
                    chosen_dim = dim
                    best_score = dim_score
        else:
            raise ValueError("Unknown splitting method!")
        return chosen_dim

    def get_zono_lb(self, net, s1):
        z1 = ZonoTransformer(s1)
        z1 = parse.get_transformer(z1, net, self)
        lb1, _, _ = z1.compute_lb(complete=True)
        if lb1 is None:
            lb1 = 0
        return lb1
