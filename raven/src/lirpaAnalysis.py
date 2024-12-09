import torch

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import sys
sys.path.append('nn-proof-transfer/nn_pruning')
sys.path.append('nn-proof-transfer/nn_pruning/sparsification-result')

from src.network_conversion_helper import get_pytorch_net
from src.common import Status
from src import util
from src.util import get_linear_layers
from src.sparsification_util import get_sparsification_indices, prune_last_layer
from copy import deepcopy
from src.common import Domain
from src.common.dataset import Dataset
from auto_LiRPA.operators import BoundLinear, BoundConv, BoundRelu


class LirpaTransformer:
    def __init__(self, domain, dataset):
        """"
        prop: Property for verification
        """
        self.domain = domain
        self.dataset = dataset

        if domain == Domain.LIRPA_IBP:
            self.method = 'IBP'
        elif domain == Domain.LIRPA_CROWN_IBP:
            self.method = 'backward'
        elif domain == Domain.LIRPA_CROWN:
            self.method = 'CROWN'
        elif domain == Domain.LIRPA_CROWN_OPT:
            self.method = 'CROWN-Optimized'
        elif domain == Domain.LIRPA_CROWN_FORWARD:
            self.method = 'Forward+Backward'

        self.model = None
        self.ilb = None
        self.iub = None
        self.input = None
        self.out_spec = None
        self.batch_size = None
        self.prop = None
        self.args = None
        self.sparsification_result = None

    def get_sparsification_result(self):
        return self.sparsification_result

    def build(self, net, prop, relu_mask=None):
        self.ilb = util.reshape_input(prop.input_props[-1].input_lb, self.dataset)
        self.iub = util.reshape_input(prop.input_props[-1].input_ub, self.dataset)
        self.input = (self.ilb + self.iub) / 2
        self.batch_size = self.input.shape[0]
        self.model = BoundedModule(net, torch.empty_like(self.input), device=prop.input_props[-1].input_lb.device)
        self.out_spec = prop.out_constr.constr_mat[0].T.unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.prop = prop

    def is_linear(self, net_name):
        if 'mnist-net_256x2.onnx' in net_name:
            return True
        else:
            return False

    def get_modified_constraint_and_bias(self, pruned_linear_layer, constraint_mat):
        weight = pruned_linear_layer.weight.T
        bias = pruned_linear_layer.bias
        new_constraint = weight @ constraint_mat
        new_bias = bias @ constraint_mat
        return new_constraint, new_bias

    #  Not updated update before use
    def extract_abstract_features(self, net, zero_feature_indices, nonzero_feture_indices, 
                                    final_layer):        
        initial_sparsity = nonzero_feture_indices.size()[0]
        pruned_feture_count = 0
        l = 0
        r = initial_sparsity - 1
        # Compute the model without the last layer.
        while l <= r:
            mid = (l + r) // 2
            if mid <= 0:
                break
            # Pop the last layer from the net
            _ = net.pop()
            final_layer_copy = deepcopy(final_layer)
            indices_to_prune = nonzero_feture_indices[:mid]
            prune_last_layer(final_layer_copy.weight, indices_to_prune)
            net.append(final_layer_copy)
            pytorch_model = get_pytorch_net(model=net, remove_last_layer=False,
                                                        all_linear=self.is_linear(self.args.net))
            self.build(net=pytorch_model, prop=self.prop)
            lb = self.compute_lb(C=self.out_spec)
            if lb >= 0:
                pruned_feture_count = max(pruned_feture_count, mid)
                l = mid + 1
            else:
                r = mid - 1
        optimal_sparsity = initial_sparsity - pruned_feture_count
        return optimal_sparsity


    def compute_sparse_features(self, net, f_lb, f_ub):
        linear_layers = get_linear_layers(net)
        final_layer = linear_layers[-1]
        nozero_count, zero_feature_indices, nonzero_feature_indices = get_sparsification_indices(f_lb, 
                                        f_ub, final_layer.weight, self.prop.out_constr.constr_mat[0])
        optimal_sparsity = self.extract_abstract_features(net=net, zero_feature_indices=zero_feature_indices,
                                                           nonzero_feture_indices=nonzero_feature_indices,
                                                           final_layer=final_layer)
        self.sparsification_result = {}
        self.sparsification_result["Initial sparsity"] = nozero_count
        self.sparsification_result["Optimal Sparsity"] = optimal_sparsity
        self.sparsification_result["zero indices"] = zero_feature_indices
        self.sparsification_result["Indices prune"] = nonzero_feature_indices[:(nozero_count - optimal_sparsity)]
        self.sparsification_result["Remaining indices"] = nonzero_feature_indices[(nozero_count - optimal_sparsity):]
        return



    def verify_property(self, prop, args):
        net = util.get_net(args.net, args.dataset)
        self.args = args
        self.sparsification_result = None
        # Temporary fix for avoiding sparsification.
        pytorch_model = get_pytorch_net(model=net, remove_last_layer=False, all_linear=self.is_linear(args.net))
        if args.enable_sparsification:
            pytorch_model_wt_last_layer = get_pytorch_net(model=net, remove_last_layer=True, all_linear=self.is_linear(args.net))
        else:
            pytorch_model_wt_last_layer = None
        f_lb = None
        f_ub = None      
        if args.enable_sparsification:
            self.build(net=pytorch_model_wt_last_layer, prop=prop)
            f_lb, f_ub = self.compute_lb_ub(C=None)
            f_lb = torch.squeeze(f_lb)
            f_ub = torch.squeeze(f_ub)
        
        self.build(net=pytorch_model, prop=prop)
        lb = self.compute_lb(C=self.out_spec)
        verification_result = Status.UNKNOWN
        print("Method", self.method)
        print("Lower bound", lb)
        if lb >= 0:
            verification_result = Status.VERIFIED
        else:
            return verification_result
        if args.enable_sparsification and verification_result == Status.VERIFIED:
            self.compute_sparse_features(net=net, f_lb=f_lb, f_ub=f_ub)
        return verification_result

    
    def compute_lb_ub(self, C=None):
        ptb = PerturbationLpNorm(x_L=self.ilb, x_U=self.iub)
        lirpa_input_spec = BoundedTensor(self.input, ptb)
        olb, oub = self.model.compute_bounds(x=(lirpa_input_spec,), method=self.method, C=C)
        return olb, oub

    def compute_lb(self, C=None, complete=False):
        ptb = PerturbationLpNorm(x_L=self.ilb, x_U=self.iub)
        lirpa_input_spec = BoundedTensor(self.input, ptb)
        olb, _ = self.model.compute_bounds(x=(lirpa_input_spec,), method=self.method, C=C)
        olb = olb + self.prop.out_constr.constr_mat[1]

        if self.prop.input_props[-1].is_conjunctive():
            lb = torch.min(olb, dim=1).values
        else:
            lb = torch.max(olb, dim=1).values

        if complete:
            return lb, True, None
        else:
            return lb

    def get_all_bounds(self):
        lbs, ubs = [], []
        lbs.append(self.ilb)
        ubs.append(self.iub)
        for node_name, node in self.model._modules.items():
            if type(node) in [BoundLinear, BoundConv] and node_name != 'last_final_node':
                lbs.append(node.lower)
                lbs.append(torch.clamp(node.lower, min=0))
                ubs.append(node.upper)
                ubs.append(torch.clamp(node.upper, min=0))
        return lbs, ubs