import torch
from torch.nn import functional as F
from raven.src.baseline_uap_verifier_res import BaselineVerifierRes
from raven.src.sparsification_util import get_sparsification_indices, prune_last_layer
from copy import deepcopy

device = 'cpu'


class ZonoTransformer:
    def __init__(self, prop, cof_constrain=None, bias_constrain=None, complete=False):
        """
        ilb: the lower bound for the input variables
        iub: the upper bound for the input variables
        """
        self.size = prop.get_input_size()
        self.prop = prop
        self.ilb = prop.input_lb
        self.iub = prop.input_ub
        self.complete = complete
        # A map that keeps tracks of the scaling factor of the perturbation
        # bound for each index. Currently the perturbation bound is only defined
        # for the final two layers.
        self.perturbation_scaling = {}

        # Following fields are used for complete verification
        self.complete = complete
        self.map_for_noise_indices = {}

        if self.size == 784:
            self.shape = (1, 28, 28)
        elif self.size == 3072:
            self.shape = (3, 32, 32)
        elif self.size == 2:
            # For debug network
            self.shape = (1, 1, 2)

        self.ilb = self.ilb.to(device)
        self.iub = self.iub.to(device)

        center = (self.ilb + self.iub) / 2
        self.unstable_relus = []
        self.active_relus = []
        self.inactive_relus = []        
        noise_ind = self.get_noise_indices()

        cof = ((self.iub - self.ilb) / 2 * torch.eye(self.size))[noise_ind]

        self.centers = []
        self.cofs = []
        self.linear_centers = []
        self.linear_coefs = []
        self.log_domain_filename = './debug_logs/deepz_debug_log.txt'

        # In this case we don't multiply the constr mat with
        # the final layer weight matrix
        self.final_layer_without_constr_center = None
        self.final_layer_without_constr_coef = None

        self.set_zono(center, cof)


    
    def populate_baseline_verifier_result(self):
        final_lb = self.compute_lb()
        layer_lbs, layer_ubs = self.get_all_linear_bounds_wt_constraints()
        layer_lbs.append(self.ilb)
        layer_ubs.append(self.iub)
        coef, center = self.final_coef_center()
        return BaselineVerifierRes(input=self.prop.input, layer_lbs=layer_lbs, layer_ubs=layer_ubs, final_lb=final_lb,
                                   zono_center=center, zono_coef=coef)


    def get_noise_indices(self):
        num_eps = 1e-7
        noise_ind = torch.where(self.iub > (self.ilb + num_eps))
        if noise_ind[0].size() == 0:
            # add one dummy index in case there is no perturbation
            noise_ind = torch.tensor([0]).to(device)
        for i in range(len(noise_ind[0])):
            self.map_for_noise_indices[i] = noise_ind[0][i].item()
        return noise_ind

    def final_coef_center(self):
        center = self.centers[-1]
        coef = self.cofs[-1]
        return coef, center



    def compute_lb(self, adv_label=None, complete=False, center=None, cof=None):
        """
        return the lower bound for the variables of the current layer
        """
        if center is None or cof is None:
            center = self.centers[-1]
            cof = self.cofs[-1]
        
        if complete:
            cof = cof[:, adv_label]
            cof_abs = torch.sum(torch.abs(cof), dim=0)
            lb = center[adv_label] - cof_abs
            sz = len(self.ilb)
            signs = (cof[:sz] > 0).to(device)
            if self.prop.is_conjunctive():
                lb = torch.min(lb)
            else:
                lb = torch.max(lb)
            return lb, True, None
        else:
            cof_abs = torch.sum(torch.abs(cof), dim=0)
            lb = center - cof_abs
            return lb

    def compute_ub(self, test=True):
        """
        return the upper bound for the variables of the current layer
        """
        center = self.centers[-1]
        cof = self.cofs[-1]

        cof_abs = torch.sum(torch.abs(cof), dim=0)

        ub = center + cof_abs

        return ub

    def bound(self):
        # This can be little faster by reusing the computation
        center = self.centers[-1]
        cof = self.cofs[-1]

        cof_abs = torch.sum(torch.abs(cof), dim=0)

        lb = center - cof_abs
        ub = center + cof_abs

        return lb, ub

    def get_zono(self):
        return self.centers[-1], self.cofs[-1]

    def set_zono(self, center, cof):
        self.centers.append(center)
        self.cofs.append(cof)

    def set_linear_zono(self, center, coef):
        self.linear_centers.append(center)
        self.linear_coefs.append(coef)

    def get_all_linear_bounds_wt_constraints(self):
        lbs = []
        ubs = []

        for i in range(len(self.linear_centers)):
            center = self.linear_centers[i]
            coef = self.linear_coefs[i]

            coef_abs = torch.sum(torch.abs(coef), dim=0)

            lb = center - coef_abs
            ub = center + coef_abs

            lbs.append(lb)
            ubs.append(ub)

        # update bounds without constraints.
        lbs.pop()
        ubs.pop()
        coef_abs = torch.sum(torch.abs(self.final_layer_without_constr_coef), dim=0)
        lb = self.final_layer_without_constr_center - coef_abs
        ub = self.final_layer_without_constr_center + coef_abs
        lbs.append(lb)
        ubs.append(ub)
        return lbs, ubs


    def get_all_bounds(self):
        lbs = []
        ubs = []

        for i in range(len(self.centers)):
            center = self.centers[i]
            cof = self.cofs[i]

            cof_abs = torch.sum(torch.abs(cof), dim=0)

            lb = center - cof_abs
            ub = center + cof_abs

            lbs.append(lb)
            ubs.append(ub)

        return lbs, ubs
    


    def get_layer_bound(self, index):
        lbs, ubs = self.get_all_bounds()
        try:
            return lbs[index], ubs[index]
        except:
            raise ValueError("Index out of bound")

    def get_active_relu_list(self):
        return self.active_relus

    def get_inactive_relu_list(self):
        return self.inactive_relus

    # Find the scaling factor to scale perturbation bound.
    def get_perturbation_scaling(self, layer_index):
        if layer_index not in [-1, -2]:
            raise ValueError("Perturbation scaling is not implemented for any layer other than last two layers")
        if layer_index not in self.perturbation_scaling.keys():
            return None
        else:
            return self.perturbation_scaling[layer_index]
    

    # Populate the scaling factor for perturbation for different
    # index.
    def populate_perturbation_scaling_factor(self, last_layer_wt, output_specification_mat):
        if output_specification_mat is None:
            self.perturbation_scaling[-1] = None
        else:
            # self.perturbation_scaling[-1] = torch.max(torch.norm(output_specification_mat, dim=0))
            self.perturbation_scaling[-1] = 1.0
        if last_layer_wt is None:
            self.perturbation_scaling[-2] = None
        else:
            self.perturbation_scaling[-2] = torch.max(torch.norm(last_layer_wt, dim=0))


    def verify_property_with_pruned_layer(self, pruned_final_layer, adv_label, complete):
        prev_center, prev_coefficent = self.centers[-2], self.cofs[-2]
        weight = pruned_final_layer.weight.T
        bias = pruned_final_layer.bias
        weight = weight @ self.prop.output_constr_mat()
        bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()
        center = prev_center @ weight + bias
        cof = prev_coefficent @ weight
        lb, _, _ = self.compute_lb(adv_label=adv_label, complete=complete, center=center, cof=cof)
        if lb is not None and torch.all(lb >= 0):
            return True
        else:
            return False

    
    def extract_abstract_features(self, zero_feature_indices, nonzero_feture_indices, 
                                    final_layer, adv_label, complete):
        prune_last_layer(final_layer.weight, zero_feature_indices)
        initial_sparsity = nonzero_feture_indices.size()[0]
        pruned_feture_count = 0
        l = 0
        r = initial_sparsity - 1
        while l <= r:
            mid = (l + r) // 2
            if mid <= 0:
                break
            final_layer_copy = deepcopy(final_layer)
            indices_to_prune = nonzero_feture_indices[:mid]
            prune_last_layer(final_layer_copy.weight, indices_to_prune)
            verification_res = self.verify_property_with_pruned_layer(final_layer_copy, adv_label, complete)
            if verification_res:
                pruned_feture_count = max(pruned_feture_count, mid)
                l = mid + 1
            else:
                r = mid - 1
        optimal_sparsity = initial_sparsity - pruned_feture_count
        return optimal_sparsity



    def compute_sparsification(self, final_layer, adv_label, complete):
        f_lbs, f_ubs = self.get_layer_bound(-2)
        nozero_count, zero_feature_indices, nonzero_feature_indices = get_sparsification_indices(f_lbs, 
                                                f_ubs, final_layer.weight, self.prop.output_constr_mat())
        
        optimal_sparsity = self.extract_abstract_features(zero_feature_indices, nonzero_feature_indices, 
                                                final_layer, adv_label, complete)
        sparsification_result = {}
        sparsification_result["Initial sparsity"] = nozero_count
        sparsification_result["Optimal Sparsity"] = optimal_sparsity
        sparsification_result["zero indices"] = zero_feature_indices
        sparsification_result["Indices prune"] = nonzero_feature_indices[:(nozero_count - optimal_sparsity)]
        sparsification_result["Remaining indices"] = nonzero_feature_indices[(nozero_count - optimal_sparsity):]
        return sparsification_result



    def handle_normalization(self, layer):
        '''
        only change the lower/upper bound of the input variables
        '''
        return
        # mean = layer.mean.view((1))
        # sigma = layer.sigma.view((1))
        #
        # prev_cent, prev_cof = self.get_zono()
        #
        # center = (prev_cent - mean) / sigma
        # cof = prev_cof / sigma
        #
        # self.set_zono(center, cof)
        #
        # return self

    def handle_addition(self, layer, last_layer=False):
        """
        handle addition layer
        """
        bias = layer.bias
        if last_layer:
            bias = bias @ self.prop.output_constr_mat()

        prev_cent, prev_cof = self.get_zono()

        center = prev_cent + bias
        cof = prev_cof

        self.set_zono(center, cof)
        return self

    def handle_linear(self, layer, last_layer=False):
        """
        handle linear layer
        """
        weight = layer.weight.T
        bias = layer.bias
        if last_layer:
            org_weight = weight
            org_bias = bias
            weight = weight @ self.prop.output_constr_mat()
            bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()
            self.populate_perturbation_scaling_factor(weight, self.prop.output_constr_mat())
            # print("output bias", bias)
        self.shape = (1, weight.shape[1])
        self.size = weight.shape[1]

        prev_cent, prev_cof = self.get_zono()

        center = prev_cent @ weight + bias
        cof = prev_cof @ weight

        if last_layer:
            self.final_layer_without_constr_center = prev_cent @ org_weight + org_bias
            self.final_layer_without_constr_coef = prev_cof @ org_weight

        self.set_zono(center, cof)
        self.set_linear_zono(center=center, coef=cof)
        return self

    def handle_conv2d(self, layer):
        """
        handle conv2d layer
        first transform it to linear matrix
        then use absmul func
        """
        weight = layer.weight
        bias = layer.bias
        num_kernel = weight.shape[0]

        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding

        shape = self.shape

        input_h, input_w = shape[1:]

        ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        self.shape = (num_kernel, output_h, output_w)
        self.size = num_kernel * output_h * output_w

        prev_cent, prev_cof = self.get_zono()

        prev_cent = prev_cent.reshape(1, shape[0], input_h, input_w)
        prev_cof = prev_cof.reshape(-1, shape[0], input_h, input_w)

        center = F.conv2d(prev_cent, weight, padding=layer.padding, stride=layer.stride, bias=bias).flatten()

        num_eps = prev_cof.shape[0]
        cof = F.conv2d(prev_cof, weight, padding=layer.padding, stride=layer.stride).reshape(num_eps, -1)

        self.set_zono(center, cof)
        self.set_linear_zono(center=center, coef=cof)

        return self

    def handle_relu(self, layer, optimize=True, relu_mask=None):
        """
        handle relu func
        """
        size = self.size

        prev_cent, prev_cof = self.get_zono()
        lb, ub = self.bound()

        layer_no = len(self.unstable_relus)
        self.unstable_relus.append(torch.where(torch.logical_and(ub >= 0, lb <= 0))[0].tolist())

        num_eps = 1e-7
        lmbda = torch.div(ub, ub - lb + num_eps)
        mu = -(lb / 2) * lmbda

        active_relus = (lb > 0)
        passive_relus = (ub <= 0)
        ambiguous_relus = (~active_relus) & (~passive_relus)

        self.active_relus.append(torch.where(active_relus)[0].tolist())
        self.inactive_relus.append(torch.where(passive_relus)[0].tolist())

        if self.complete:
            # Store the map from (unstable relu index -> index of the added noise)
            prev_error_terms = prev_cof.shape[0]
            unstable_relu_indices = torch.where(ambiguous_relus)[0]

            for i, index in enumerate(unstable_relu_indices):
                index_of_unstable_relu = prev_error_terms + i
                self.map_for_noise_indices[index_of_unstable_relu] = (layer_no, index.item())

            # Figure out how these should be used
            c1_decision = torch.zeros(size, dtype=torch.bool)
            c2_decision = torch.zeros(size, dtype=torch.bool)

            if relu_mask is not None:
                for relu in relu_mask.keys():
                    if relu[0] == layer_no:
                        if ambiguous_relus[relu[1]]:
                            if relu_mask[relu] == 1:
                                c1_decision[relu[1]] = 1
                            elif relu_mask[relu] == -1:
                                c2_decision[relu[1]] = 1

            ambiguous_relus = ambiguous_relus & (~c1_decision) & (~c2_decision)
            c1_mu = c1_decision*ub/2
            c2_mu = c2_decision*lb/2

        mult_fact = torch.ones(size, dtype=torch.bool)
        # mult_fact should have 1 at active relus, 0 at passive relus and lambda at ambiguous_relus
        mult_fact = mult_fact * (active_relus + ambiguous_relus * lmbda)

        if self.complete:
            new_noise_cofs = torch.diag(mu * ambiguous_relus + c1_mu + c2_mu)
        else:
            new_noise_cofs = torch.diag(mu * ambiguous_relus)

        non_empty_mask = new_noise_cofs.abs().sum(dim=0).bool()
        new_noise_cofs = new_noise_cofs[non_empty_mask, :]
        cof = torch.cat([mult_fact * prev_cof, new_noise_cofs])
        if self.complete:
            center = prev_cent * active_relus + (lmbda * prev_cent + mu) * ambiguous_relus + c1_mu + c2_mu
        else:
            center = prev_cent * active_relus + (lmbda * prev_cent + mu) * ambiguous_relus

        self.set_zono(center, cof)
        return self

    def verify_robustness(self, y, true_label):
        pass
