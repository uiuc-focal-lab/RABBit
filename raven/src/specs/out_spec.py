import torch

from enum import Enum


class OutSpecType(Enum):
    LOCAL_ROBUST = 1
    GLOBAL = 2


# Output constraint is represented as (A^T)*Y + B >= 0
# Where constr_mat = (A, B)
class Constraint:
    def __init__(self, constr_type, is_conjunctive=True, constr_mat=None, 
                 label=None, sink_label=None, debug_mode=False):
        self.constr_type = constr_type
        self.label = label
        self.constr_mat = constr_mat
        self.is_conjunctive = is_conjunctive
        mat = None
        if debug_mode == True:
            if self.label == 0:
                mat = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
            else:
                mat = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
            self.constr_mat = (mat, 0)
        if constr_type == OutSpecType.LOCAL_ROBUST and debug_mode == False:
            if label is not None:
                mat = create_out_constr_matrix(label) 
            elif sink_label is not None:
                mat = create_out_targeted_uap_matrix(sink_label)
            else:
                raise ValueError("Label or Sink Label has to be not None")
            self.constr_mat = (mat, 0)
            
def create_out_constr_matrix(label):
    n_classes = 10
    mat = torch.zeros(size=(n_classes, n_classes - 1))
    ground_truth = label.unsqueeze(0).unsqueeze(0).type(torch.int64)
    target_label = torch.zeros(size=(1, n_classes - 1))
    for j in range(n_classes - 1):
        target_label[0, j] = (label + j + 1) % n_classes

    target_label = target_label.type(torch.int64)

    mat[label, :] = 1
    mat.scatter_(dim=0, index=ground_truth, value=1.0)
    mat.scatter_(dim=0, index=target_label, value=-1.0)
    return mat

def create_out_targeted_uap_matrix(sink_label):
    n_classes = 10
    mat = torch.zeros(size=(n_classes, n_classes - 1))
    ground_truth = sink_label.unsqueeze(0).unsqueeze(0).type(torch.int64)
    target_label = torch.zeros(size=(1, n_classes - 1))
    for j in range(n_classes - 1):
        target_label[0, j] = (sink_label + j + 1) % n_classes

    target_label = target_label.type(torch.int64)

    mat[sink_label, :] = -1
    mat.scatter_(dim=0, index=ground_truth, value=-1.0)
    mat.scatter_(dim=0, index=target_label, value=1.0)
    return mat

# This was intended for output space splitting
# class Outspec:
#     def __init__(self, lb_min, lb_max):
#         self.lb_min = lb_min
#         self.lb_max = lb_max
#         self.status = Status.UNKNOWN
