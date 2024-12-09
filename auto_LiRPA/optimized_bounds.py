import time
import os
import warnings
from collections import OrderedDict
from contextlib import ExitStack
from itertools import combinations

import torch
from torch import optim
from .cuda_utils import double2float
from .utils import logger
from time import time


def _set_alpha(optimizable_activations, parameters, alphas, lr):
    """
    Set best_alphas, alphas and parameters list
    """
    for node in optimizable_activations:
        alphas.extend(list(node.alpha.values()))
        node.opt_start()
    # Alpha has shape (2, output_shape, batch_dim, node_shape)
    parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
    # best_alpha is a dictionary of dictionary. Each key is the alpha variable
    # for one relu layer, and each value is a dictionary contains all relu
    # layers after that layer as keys.
    best_alphas = OrderedDict()
    for m in optimizable_activations:
        best_alphas[m.name] = {}
        for alpha_m in m.alpha:
            best_alphas[m.name][alpha_m] = m.alpha[alpha_m].detach().clone()
            # We will directly replace the dictionary for each relu layer after
            # optimization, so the saved alpha might not have require_grad=True.
            m.alpha[alpha_m].requires_grad_()

    return best_alphas


def _set_beta(
        self, relus, optimizable_activations, single_node_split,
        enable_opt_interm_bounds, betas, opt_coeffs, parameters,
        lr_coeffs, opt_bias, lr_beta, lr_cut_beta, cutter, dense_coeffs_mask):
    """
    Set betas, best_betas, coeffs, dense_coeffs_mask, best_coeffs, biases
    and best_biases.
    """
    coeffs = best_coeffs = biases = best_biases = None
    if len(relus) != len(optimizable_activations):
        warnings.warn(
            'Only relu split is supported so far, this model contains other '
            'optimizable activations that may not apply split.')

    if single_node_split:
        for node in relus:
            if enable_opt_interm_bounds and node.sparse_beta is not None:
                for key in node.sparse_beta.keys():
                    if node.sparse_beta[key] is not None:
                        betas.append(node.sparse_beta[key])
            else:
                if node.sparse_beta is not None:
                    betas.append(node.sparse_beta)
    else:
        betas = self.beta_params + self.single_beta_params
        if opt_coeffs:
            coeffs = [dense_coeffs['dense']
                for dense_coeffs in self.split_dense_coeffs_params
            ] + self.coeffs_params
            dense_coeffs_mask = [dense_coeffs['mask']
                for dense_coeffs in self.split_dense_coeffs_params]
            parameters.append({'params': coeffs, 'lr': lr_coeffs})
            best_coeffs = [coeff.detach().clone() for coeff in coeffs]
        if opt_bias:
            biases = self.bias_params
            parameters.append({'params': biases, 'lr': lr_coeffs})
            best_biases = [bias.detach().clone() for bias in biases]

    # Beta has shape (batch, max_splits_per_layer)
    parameters.append({'params': betas, 'lr': lr_beta, 'batch_dim': 0})

    if self.cut_used:
        # also need to optimize cut betas
        parameters.append({'params': self.cut_beta_params,
                          'lr': lr_cut_beta, 'batch_dim': 0})
        betas = betas + self.cut_beta_params

    if enable_opt_interm_bounds and betas:
        best_betas = OrderedDict()
        for m in optimizable_activations:
            best_betas[m.name] = {}
            for beta_m, beta in m.sparse_beta.items():
                best_betas[m.name][beta_m] = beta.detach().clone()
        if self.cut_used:
            best_betas['cut'] = []
            for general_betas in self.cut_beta_params:
                best_betas['cut'].append(general_betas.detach().clone())
    else:
        best_betas = [b.detach().clone() for b in betas]

    if self.cut_used and getattr(cutter, 'opt', False):
        parameters.append(cutter.get_parameters())

    return (
        betas, best_betas, coeffs, dense_coeffs_mask, best_coeffs, biases,
        best_biases)


def _save_ret_first_time(bounds, fill_value, x, best_ret):
    """
    Save results at the first iteration to best_ret
    """
    if bounds is not None:
        best_bounds = torch.full_like(
            bounds, fill_value=fill_value, device=x[0].device, dtype=x[0].dtype)
    else:
        best_bounds = None

    if bounds is not None:
        best_ret.append(bounds.detach().clone())
    else:
        best_ret.append(None)

    return best_bounds


@torch.no_grad()
def _get_preserve_mask(
        decision_thresh, ret_l, preserve_mask, multi_spec_keep_func):
    """
    Get preserve mask by decision_thresh to filter out the satisfied bounds.
    """
    if (isinstance(decision_thresh, torch.Tensor)
            and decision_thresh.numel() > 1):
        if decision_thresh.shape[-1] == 1:
            now_preserve_mask = (
                ret_l <= decision_thresh[preserve_mask]
            ).view(-1).nonzero().view(-1)
        else:
            now_preserve_mask = multi_spec_keep_func(
                ret_l <= decision_thresh[preserve_mask]).nonzero().view(-1)
    else:
        if decision_thresh.shape[-1] == 1:
            now_preserve_mask = (
                ret_l <= decision_thresh).view(-1).nonzero().view(-1)
        else:
            now_preserve_mask = multi_spec_keep_func(
                ret_l <= decision_thresh).nonzero().view(-1)

    return now_preserve_mask


def _recover_bounds_to_full_batch(
        ret, decision_thresh, epsilon_over_decision_thresh, original_size,
        preserve_mask, loss_reduction_func):
    """
    Recover lower and upper bounds to full batch size so that later we can
    directly update using the full batch size of l and u.
    """
    if ret is not None:
        if (isinstance(decision_thresh, torch.Tensor)
                and decision_thresh.numel() > 1):
            full_ret = (decision_thresh.clone().to(ret.device).type(ret.dtype)
                        + epsilon_over_decision_thresh)
        else:
            num_decision_thresh = decision_thresh
            if isinstance(num_decision_thresh, torch.Tensor):
                num_decision_thresh = num_decision_thresh.item()
            full_ret = torch.full(
                (original_size,) + tuple(ret.shape[1:]),
                fill_value=num_decision_thresh + epsilon_over_decision_thresh,
                device=ret.device, dtype=ret.dtype)
        full_ret[preserve_mask] = ret
        if full_ret.shape[1] > 1:
            full_reduced_ret = loss_reduction_func(full_ret)
        else:
            full_reduced_ret = full_ret
    else:
        full_ret = full_reduced_ret = None

    return full_ret, full_reduced_ret


@torch.no_grad()
def _prune_bounds_by_mask(
        now_preserve_mask, decision_thresh, ret_l, ret_u, ret, preserve_mask,
        epsilon_over_decision_thresh, original_size, loss_reduction_func,
        beta, intermediate_beta_enabled,
        fix_intermediate_layer_bounds, intermediate_layer_bounds,
        aux_reference_bounds, partial_intermediate_layer_bounds,
        pre_prune_size):
    """
    Prune bounds by given now_preserve_mask.
    """
    full_ret_l, full_l = _recover_bounds_to_full_batch(
        ret_l, decision_thresh, epsilon_over_decision_thresh,
        original_size, preserve_mask, loss_reduction_func)

    full_ret_u, full_u = _recover_bounds_to_full_batch(
        ret_u, decision_thresh, epsilon_over_decision_thresh,
        original_size, preserve_mask, loss_reduction_func)

    full_ret = (full_ret_l, full_ret_u) + ret[2:]

    if beta and intermediate_beta_enabled:
        # prune the partial_intermediate_layer_bounds
        interval_to_prune = partial_intermediate_layer_bounds
    elif fix_intermediate_layer_bounds:
        interval_to_prune = intermediate_layer_bounds
    else:
        interval_to_prune = None
    if interval_to_prune is not None:
        for k, v in interval_to_prune.items():
            interm_interval_l, interm_interval_r = v[0], v[1]
            if interm_interval_l.shape[0] == pre_prune_size:
                # the first dim is batch size and matches preserve mask
                interm_interval_l = interm_interval_l[now_preserve_mask]
            if interm_interval_r.shape[0] == pre_prune_size:
                # the first dim is batch size and matches preserve mask
                interm_interval_r = interm_interval_r[now_preserve_mask]
            interval_to_prune[k] = [interm_interval_l, interm_interval_r]

    if aux_reference_bounds is not None:
        for k in aux_reference_bounds:
            aux_ref_l, aux_ref_r = aux_reference_bounds[k]
            if aux_ref_l.shape[0] == pre_prune_size:
                # the first dim is batch size and matches the preserve mask
                aux_ref_l = aux_ref_l[now_preserve_mask]
            if aux_ref_r.shape[0] == pre_prune_size:
                # the first dim is batch size and matches the preserve mask
                aux_ref_r = aux_ref_r[now_preserve_mask]
            aux_reference_bounds[k] = [aux_ref_l, aux_ref_r]

    # update the global mask here for possible next iteration
    preserve_mask_next = preserve_mask[now_preserve_mask]

    return full_l, full_ret_l, full_u, full_ret_u, full_ret, preserve_mask_next


@torch.no_grad()
def _prune_x(x, now_preserve_mask):
    """
    Prune x by given now_preserve_mask.
    """
    x = list(x)
    pre_prune_size = x[0].shape[0]
    x[0].data = x[0][now_preserve_mask].data
    if hasattr(x[0], 'ptb'):
        if x[0].ptb.x_L is not None:
            x[0].ptb.x_L = x[0].ptb.x_L[now_preserve_mask]
        if x[0].ptb.x_U is not None:
            x[0].ptb.x_U = x[0].ptb.x_U[now_preserve_mask]
    x = tuple(x)

    return x, pre_prune_size


def _to_float64(self, C, x, aux_reference_bounds, intermediate_layer_bounds):
    """
    Transfer variables to float64 only in the last iteration to help alleviate
    floating point error.
    """
    self.to(torch.float64)
    C = C.to(torch.float64)
    x = self._to(x, torch.float64)
    # best_intermediate_bounds is linked to aux_reference_bounds!
    # we only need call .to() for one of them
    self._to(aux_reference_bounds, torch.float64, inplace=True)
    intermediate_layer_bounds = self._to(
        intermediate_layer_bounds, torch.float64)

    return C, x, intermediate_layer_bounds


def _to_default_dtype(
        self, x, total_loss, full_ret, ret, best_intermediate_bounds, return_A):
    """
    Switch back to default precision from float64 typically to adapt to
    afterwards operations.
    """
    total_loss = total_loss.to(torch.get_default_dtype())
    self.to(torch.get_default_dtype())
    x[0].to(torch.get_default_dtype())
    full_ret = list(full_ret)
    if isinstance(ret[0], torch.Tensor):
        # round down lower bound
        full_ret[0] = double2float(full_ret[0], 'down')
    if isinstance(ret[1], torch.Tensor):
        # round up upper bound
        full_ret[1] = double2float(full_ret[1], 'up')
    for _k, _v in best_intermediate_bounds.items():
        _v[0] = double2float(_v[0], 'down')
        _v[1] = double2float(_v[1], 'up')
        best_intermediate_bounds[_k] = _v
    if return_A:
        full_ret[2] = self._to(full_ret[2], torch.get_default_dtype())

    return total_loss, x, full_ret


def _get_idx_mask(idx, full_ret_bound, best_ret_bound):
    """Get index for improved elements."""
    assert idx in [0, 1], (
        '0 means updating lower bound, 1 means updating upper bound')
    if idx == 0:
        idx_mask = (full_ret_bound > best_ret_bound).any(dim=1).view(-1)
    else:
        idx_mask = (full_ret_bound < best_ret_bound).any(dim=1).view(-1)

    improved_idx = None
    if idx_mask.any():
        # we only pick up the results improved in a batch
        improved_idx = idx_mask.nonzero(as_tuple=True)[0]
    return idx_mask, improved_idx


def _update_best_ret(
        full_ret_bound, best_ret_bound, full_ret, best_ret, need_update, idx):
    """Update best_ret_bound and best_ret by comparing with new results."""
    assert idx in [0, 1], (
        '0 means updating lower bound, 1 means updating upper bound')
    idx_mask, improved_idx = _get_idx_mask(idx, full_ret_bound, best_ret_bound)

    if improved_idx is not None:
        need_update = True
        if idx == 0:
            best_ret_bound[improved_idx] = torch.maximum(
                full_ret_bound[improved_idx], best_ret_bound[improved_idx])
            if full_ret[idx] is not None:
                best_ret[idx][improved_idx] = torch.maximum(
                    full_ret[idx][improved_idx], best_ret[idx][improved_idx])
        else:
            best_ret_bound[improved_idx] = torch.minimum(
                full_ret_bound[improved_idx], best_ret_bound[improved_idx])
            if full_ret[idx] is not None:
                best_ret[idx][improved_idx] = torch.minimum(
                    full_ret[idx][improved_idx], best_ret[idx][improved_idx])

    return best_ret_bound, best_ret, idx_mask, improved_idx, need_update


def _update_optimizable_activations(
        optimizable_activations, intermediate_layer_bounds,
        fix_intermediate_layer_bounds, best_intermediate_bounds,
        reference_idx, idx, alpha, best_alphas):
    """
    Update bounds and alpha of optimizable_activations.
    """
    for node in optimizable_activations:
        # Update best intermediate layer bounds only when they are optimized.
        # If they are already fixed in intermediate_layer_bounds, then do
        # nothing.
        # print(f'intermediate_layer_bounds {intermediate_layer_bounds}')
        if (intermediate_layer_bounds is None
                or node.inputs[0].name not in intermediate_layer_bounds
                or not fix_intermediate_layer_bounds):
            best_intermediate_bounds[node.name][0][idx] = torch.max(
                best_intermediate_bounds[node.name][0][idx],
                node.inputs[0].lower[reference_idx])
            best_intermediate_bounds[node.name][1][idx] = torch.min(
                best_intermediate_bounds[node.name][1][idx],
                node.inputs[0].upper[reference_idx])

        if alpha:
            # Each alpha has shape (2, output_shape, batch, *shape) for ReLU.
            # For other activation function this can be different.
            for alpha_m in node.alpha:
                if node.alpha_batch_dim == 2:
                    best_alphas[node.name][alpha_m][:, :,
                        idx] = node.alpha[alpha_m][:, :, idx]
                elif node.alpha_batch_dim == 3:
                    best_alphas[node.name][alpha_m][:, :, :,
                        idx] = node.alpha[alpha_m][:, :, :, idx]
                else:
                    raise ValueError(
                        f'alpha_batch_dim={node.alpha_batch_dim} must be set '
                        'to 2 or 3 in BoundOptimizableActivation')


def _update_best_beta(
        self, enable_opt_interm_bounds, betas, optimizable_activations,
        best_betas, idx):
    """
    Update best beta by given idx.
    """
    if enable_opt_interm_bounds and betas:
        for node in optimizable_activations:
            for key in node.sparse_beta.keys():
                best_betas[node.name][key] = (
                    node.sparse_beta[key].detach().clone())
        if self.cut_used:
            for gbidx, general_betas in enumerate(self.cut_beta_params):
                best_betas['cut'][gbidx] = general_betas.detach().clone()
    else:
        if self.cut_used:
            regular_beta_length = len(betas) - len(self.cut_beta_params)
            for beta_idx in range(regular_beta_length):
                # regular beta crown betas
                best_betas[beta_idx][idx] = betas[beta_idx][idx]
            for cut_beta_idx in range(len(self.cut_beta_params)):
                # general cut beta crown general_betas
                best_betas[regular_beta_length + cut_beta_idx][:, :, idx,
                    :] = betas[regular_beta_length + cut_beta_idx][:, :, idx, :]
        else:
            for beta_idx in range(len(betas)):
                # regular beta crown betas
                best_betas[beta_idx][idx] = betas[beta_idx][idx]



def process_topk_coef(self, lb_coef, lambda_t):
    lb_coef_updated = lb_coef.reshape(lb_coef.shape[0], -1)
    lb_coef_updated = lb_coef_updated.reshape(lb_coef.shape[0] * lb_coef.shape[1], -1)
    lb_coef_updated = (lb_coef_updated.T * lambda_t).T
    lb_coef_updated = lb_coef_updated.reshape(lb_coef.shape[0], -1, lb_coef.shape[2]).sum(dim=1)
    return lb_coef_updated

def process_topk_bias(self, lb_bias, lambda_t):
    lb_bias_updated = lb_bias.reshape(-1)
    lb_bias_updated = lambda_t * lb_bias_updated
    lb_bias_updated = lb_bias_updated.reshape(lb_bias.shape[0], -1).sum(dim=1)
    return lb_bias_updated

def get_topk_lambda(self, lambda_t, lb):
    assert lambda_t.shape[0] == lb.shape[0] * lb.shape[1]
    temp_lb = lb.reshape(-1)
    temp_lb = temp_lb.detach()
    temp_lb = temp_lb < 0
    lambda_t = torch.sigmoid(lambda_t)
    lambda_t = temp_lb * lambda_t
    lambda_t = lambda_t.reshape(lb.shape[0], -1) 
    lambda_t = lambda_t.T / (lambda_t.sum(dim=1) + 1e-17)
    lambda_t = lambda_t.T 
    lambda_t = lambda_t.reshape(-1)
    return lambda_t    

# lb_coef: (batch_size, no_of_class - 1, input_dimension)
# lambda_t: batch_size * no_of_class - 1
# lb_bias: (batch_size, no_of_class - 1)
def get_topk_loss(self, lb_coef, lb_bias, lambda_t, lb, ptb):
    lambda_t_updated = self.get_topk_lambda(lambda_t=lambda_t, lb=lb)
    lb_coef_updated = self.process_topk_coef(lb_coef=lb_coef, lambda_t=lambda_t_updated)
    lb_bias_updated = self.process_topk_bias(lb_bias=lb_bias, lambda_t=lambda_t_updated)
    # Write the final loss concretizing with ptb
    lb_coef_updated = lb_coef_updated.reshape(lb_coef.shape[0], -1, lb_coef.shape[2])
    topk_loss = ptb.concretize(x=None, A=lb_coef_updated)
    topk_loss = topk_loss.squeeze()
    topk_loss = topk_loss + lb_bias_updated
    # import pdb; pdb.set_trace()
    print(f"topk loss {topk_loss}")
    return topk_loss

def get_final_topk_tuple(self, lb, topk, debug=False):
    assert topk is not None
    assert len(lb.shape) > 1
    unverified_indices_list = []
    max_length = 0
    empty_indices = []
    i = 0
    for t in lb:
        unverified_indices = (t < 0).nonzero().squeeze()
        if len(unverified_indices.shape) <= 0 or unverified_indices.shape[0] == 0:
            empty_indices.append(i)
            if unverified_indices.numel() == 0:
                unverified_indices_list.append(torch.tensor([], device=lb.device))
            else:
                unverified_indices_list.append(torch.tensor([unverified_indices.item()], device=lb.device))
            i += 1
            continue 
        unverified_indices_list.append(torch.tensor(list(combinations(unverified_indices, topk)), device=lb.device))
        max_length = max(max_length, unverified_indices_list[-1].shape[0])
        i += 1 
    return unverified_indices_list, max_length, empty_indices

def get_final_topk_lambda(self, lambda_t, indicator):
    lambda_t_updated = torch.sigmoid(lambda_t)
    lambda_t_updated = lambda_t_updated.reshape(indicator.shape)
    lambda_t_updated = indicator * lambda_t_updated 
    lambda_t_updated = lambda_t_updated.T / (lambda_t_updated.sum(dim=1) + 1e-17)
    lambda_t_updated = lambda_t_updated.T
    lambda_t_updated = lambda_t_updated.reshape(-1)
    return lambda_t_updated

def get_final_topk_loss(self, lb_coef, lb_bias, lambda_t, lb, ptb, topk):
    unverified_indices_list, max_length, empty_indices = self.get_final_topk_tuple(lb=lb, topk=topk)
    final_topk_loss = None
    for i in range(max_length):
        indicators = torch.zeros_like(lb, device=lb.device)
        for j, t in enumerate(unverified_indices_list):
            if t.shape[0] <= 0:
                continue
            indicators[j][t[min(t.shape[0] -1, i)]] = 1.0
        lambda_t_updated = self.get_final_topk_lambda(lambda_t, indicators)
        lb_coef_updated = self.process_topk_coef(lb_coef=lb_coef, lambda_t=lambda_t_updated)
        lb_bias_updated = self.process_topk_bias(lb_bias=lb_bias, lambda_t=lambda_t_updated)
        # Write the final loss concretizing with ptb
        lb_coef_updated = lb_coef_updated.reshape(lb_coef.shape[0], -1, lb_coef.shape[2])
        topk_loss = ptb.concretize(x=None, A=lb_coef_updated)
        topk_loss = topk_loss.squeeze()
        topk_loss = topk_loss + lb_bias_updated
        final_topk_loss = torch.min(final_topk_loss, topk_loss) if final_topk_loss is not None else topk_loss
        # import pdb; pdb.set_trace()
    bounds_from_lb = lb.sort(dim=1).values[:, topk - 1]
    if final_topk_loss is None:
        return bounds_from_lb
    return torch.max(final_topk_loss, bounds_from_lb)




def get_cross_execution_params(self, x, execution_count, 
                               parameters, lr_lambda, topk=None, spec_size=None):
    size = x[0].shape[0]
    if topk is None:
        assert (x[0].shape[0] % execution_count) == 0
        # distinct_parmeters = (x[0].shape[0] // execution_count)
        size = x[0].shape[0]
    else:
        assert spec_size is not None 
        size = x[0].shape[0] * spec_size 
    lamb = torch.ones(size, dtype=torch.float, device=x[0].device, requires_grad=True)
    parameters.append({'params': lamb, 'lr': lr_lambda})
    self.optimizable_lambda = lamb

def _compute_lb(lb_coef, lb_bias, ptb, unperturbed_images):
    lb_coef_temp = lb_coef.view(lb_coef.shape[0], lb_coef.shape[1], -1)
    center, diff = ptb.get_center_diff(unperturbed_images, lb_coef_temp)
    lb_temp = lb_coef_temp.abs().matmul(diff)
    center, lb_temp = center.squeeze(), lb_temp.squeeze()
    lower_bound = center + -1 * lb_temp + lb_bias
    # print(f'new lower_bound {lower_bound.min(dim=1)[0]}')
    return lower_bound

def extract_mininum_coef(self, lb_coef, lb_bias, ptb, unperturbed_images, indices=None):
    center, diff = ptb.get_center_diff(unperturbed_images, lb_coef)
    lb_temp = lb_coef.abs().matmul(diff)
    if center.shape[1] == 1:
        center, lb_temp = center.squeeze(dim = -1), lb_temp.squeeze(dim = -1)
    else:
        center, lb_temp = center.squeeze(), lb_temp.squeeze()
    lower_bound = center + -1 * lb_temp + lb_bias
    if indices is None:
        indices = lower_bound.min(dim=1)[1]
    # print(f'lower_bound {lower_bound.min(dim=1)[0]}')
    return lb_coef[range(len(indices)), indices, :], lb_bias[range(len(indices)), indices], center[range(len(indices)), indices], diff

def cross_execution_loss_helper(self, lb_coef_updated, lb_bias_updated,
                                execution_count, center, diff):
    lamb = torch.sigmoid(self.optimizable_lambda) + 1e-17
    assert lamb.shape[0] % execution_count == 0
    # Normalize the lambda values
    lamb = lamb.reshape(execution_count, -1)
    lamb = lamb / (lamb.sum(dim=0) + 1e-17)
    lamb = lamb.reshape(-1)
    # lamb.retain_grad()
    # print(f'lambda {self.optimizable_lambda}')
    # print(f'lambda gradient {self.optimizable_lambda.grad}')    
    lb_bias_updated = lb_bias_updated * lamb
    center = center * lamb
    lb_coef_updated = lamb * lb_coef_updated.T
    lb_coef_updated = lb_coef_updated.T
    # print(f'cen {center.shape} bias {lb_bias_updated.shape} coef {lb_coef_updated.shape}')
    assert len(lb_coef_updated.shape) == 2 
    lb_coef_updated = lb_coef_updated.view(execution_count, -1, lb_coef_updated.shape[1])
    lb_coef_updated = lb_coef_updated.sum(dim=0)
    diff = diff[0].squeeze()
    loss = lb_coef_updated.abs().matmul(diff)
    loss_bias = lb_bias_updated + center
    loss_bias = loss_bias.view(execution_count, -1)    
    loss_bias = loss_bias.sum(dim=0)
    loss = loss_bias - loss
    # print(f'cross exec loss {loss}')
    return loss


def get_cross_execution_loss(self, lb_coef, lb_bias, execution_count, ptb, unperturbed_images):
    assert lb_coef is not None
    assert lb_bias is not None
    assert lb_coef.shape[0] == lb_bias.shape[0]
    assert lb_coef.shape[0] % execution_count == 0
    assert ptb is not None
    assert unperturbed_images is not None
    # if execution_count != 2:
    #     raise ValueError(f'Currently only support pair of executions')
    lb_coef_temp = lb_coef.view(lb_coef.shape[0], lb_coef.shape[1], -1)
    lb_coef_updated, lb_bias_updated, center, diff = self.extract_mininum_coef(lb_coef_temp, lb_bias, ptb, unperturbed_images)

    loss = self.cross_execution_loss_helper(lb_coef_updated, lb_bias_updated,
                                execution_count, center, diff)
    return loss


def get_unverified_indices(approx_cross_ex_loss_tensor, lower_bound, different_executions):
    with torch.no_grad():
        unverified_indices = []
        for i in range(lower_bound.shape[0]):
            temp_indices = torch.nonzero(lower_bound[i] <= approx_cross_ex_loss_tensor[i % different_executions]).reshape(-1)
            if temp_indices.shape[0] <= 0:
                # print(f"lower_bound {lower_bound[i]} {approx_cross_ex_loss_tensor[i % different_executions]}")
                # print(f'{torch.argmin(lower_bound[i]).unsqueeze(dim=0)}')
                # import pdb; pdb.set_trace()
                temp_indices = torch.argmin(lower_bound[i]).unsqueeze(dim=0)
            unverified_indices.append(temp_indices)
        return unverified_indices

def select_inidces(list_of_indices, execution_count):
    with torch.no_grad():
        assert execution_count == len(list_of_indices)
        inidices_tuples = []
        if execution_count == 2:
            for x in list_of_indices[0]:
                for y in list_of_indices[1]:
                    inidices_tuples.append((x, y))
            return inidices_tuples
        elif execution_count == 3:
            for x in list_of_indices[0]:
                for y in list_of_indices[1]:
                    for z in list_of_indices[2]:
                        inidices_tuples.append((x, y, z))
            return inidices_tuples
        elif execution_count == 4:
            for x in list_of_indices[0]:
                for y in list_of_indices[1]:
                    for z in list_of_indices[2]:
                        for w in list_of_indices[3]:
                            inidices_tuples.append((x, y, z, w))
            return inidices_tuples
        
        elif execution_count == 5:
            for x in list_of_indices[0]:
                for y in list_of_indices[1]:
                    for z in list_of_indices[2]:
                        for w in list_of_indices[3]:
                            for a in list_of_indices[4]:
                                inidices_tuples.append((x, y, z, w, a))
            return inidices_tuples
        else:
            raise ValueError(f'Execution count {execution_count} > 5 not supported')


def get_all_indices(unverified_indices, execution_count):
    with torch.no_grad(): 
        assert len(unverified_indices) % execution_count == 0
        all_inidices = []
        max_length = 0
        different_executions = len(unverified_indices) // execution_count
        for i in range(different_executions):
            curr_indices = []
            j = i
            while j < len(unverified_indices):
                curr_indices.append(unverified_indices[j])
                j += different_executions
            all_inidices.append(select_inidces(list_of_indices=curr_indices, 
                                            execution_count=execution_count))
            max_length = max(max_length, len(all_inidices[-1]))
        
        all_ex_indices = []
        for i in range(max_length):
            indices = []
            for j in range(different_executions):
                try:
                    indices.append(all_inidices[j][min(len(all_inidices[j]) - 1, i)])
                except:
                    print(f'j {j} len {len(all_inidices[j]) - 1} i {i} final index {min(len(all_inidices[j]) - 1, i)}')
            final_indices = []
            for j in range(execution_count):
                for k in range(different_executions):
                    if type(indices[k][j]) is torch.Tensor:
                        final_indices.append(indices[k][j].item())
                    else:
                        final_indices.append(indices[k][j])
            all_ex_indices.append(final_indices)

        return all_ex_indices

def get_targeted_loss(self, lb_coef, lb_bias, target_lAs, target_biases, ptb):
    # import pdb; pdb.set_trace()
    assert lb_coef.shape == target_lAs.shape
    diff_coef = lb_coef - target_lAs
    diff_bound = ptb.concretize(x=None, A=diff_coef)
    diff_bound = diff_bound + lb_bias - target_biases
    return diff_bound
   
def get_final_cross_executional_loss(self, lb_coef, lb_bias, execution_count, lower_bound,
                                     ptb, unperturbed_images, approx_cross_ex_loss_tensor):
    # with torch.no_grad():
    assert lb_coef.shape[0] % execution_count == 0
    assert lb_bias.shape[0] % execution_count == 0
    assert lower_bound.shape[0] == lb_coef.shape[0]
    different_executions = lb_coef.shape[0] // execution_count
    unverified_indices = get_unverified_indices(approx_cross_ex_loss_tensor=approx_cross_ex_loss_tensor, 
                                                lower_bound=lower_bound, different_executions=different_executions)
    all_ex_indices = get_all_indices(unverified_indices, execution_count)
    lb_coef_temp = lb_coef.view(lb_coef.shape[0], lb_coef.shape[1], -1)
    final_loss = None
    for indices in all_ex_indices:
        indices = torch.tensor(indices, device=lb_coef_temp.device)
        lb_coef_updated, lb_bias_updated, center, diff = self.extract_mininum_coef(lb_coef_temp, lb_bias, ptb, unperturbed_images, indices)

        loss = self.cross_execution_loss_helper(lb_coef_updated, lb_bias_updated,
                                    execution_count, center, diff)
        final_loss = loss if final_loss is None else torch.min(loss, final_loss)
        
    return final_loss


def print_bound_updates(self, optimizable_activations):
    if optimizable_activations is None or len(optimizable_activations) <= 0:
        return
    with torch.no_grad():
        node = optimizable_activations[-1]
        bnd_diff = (node.inputs[0].upper - node.inputs[0].lower).sum()
        print(f'bound diff {node.inputs[0].name} {bnd_diff}')

def get_optimized_bounds(
        self, x=None, aux=None, C=None, IBP=False, forward=False,
        method='backward', bound_lower=True, bound_upper=False,
        reuse_ibp=False, return_A=False, average_A=False, final_node_name=None,
        intermediate_layer_bounds=None, reference_bounds=None,
        aux_reference_bounds=None, needed_A_dict=None, cutter=None,
        decision_thresh=None, epsilon_over_decision_thresh=1e-4,
        multiple_execution=False, execution_count=1, 
        ptb=None, unperturbed_images=None, baseline_refined_bound={},
        intermediate_bound_refinement=False, 
        always_correct_cross_execution=False,
        cross_refinement_results={}, populate_trace=False,
        target_lAs=None, target_biases=None, parent_loss=None, topk=None):
    """
    Optimize CROWN lower/upper bounds by alpha and/or beta.
    """
    start_time = time()
    opts = self.bound_opts['optimize_bound_args']
    iteration = opts['iteration']
    beta = opts['enable_beta_crown']
    alpha = opts['enable_alpha_crown']
    opt_coeffs = opts['opt_coeffs']
    opt_bias = opts['opt_bias']
    opt_choice = opts['optimizer']
    single_node_split = opts['single_node_split']
    assert single_node_split is True
    keep_best = opts['keep_best']
    fix_intermediate_layer_bounds = opts['fix_intermediate_layer_bounds']
    init_alpha = opts['init_alpha']
    lr_alpha = opts['lr_alpha']
    lr_beta = opts['lr_beta']
    lr_cut_beta = opts['lr_cut_beta']
    lr_intermediate_beta = opts['lr_intermediate_beta']
    lr_decay = opts['lr_decay']
    lr_coeffs = opts['lr_coeffs']
    loss_reduction_func = opts['loss_reduction_func']
    stop_criterion_func = opts['stop_criterion_func']
    use_float64_in_last_iteration = opts['use_float64_in_last_iteration']
    early_stop_patience = opts['early_stop_patience']
    intermediate_beta_enabled = opts['intermediate_beta']
    start_save_best = opts['start_save_best']
    multi_spec_keep_func = opts['multi_spec_keep_func']
    enable_opt_interm_bounds = self.bound_opts.get(
        'enable_opt_interm_bounds', True)
    sparse_intermediate_bounds = self.bound_opts.get(
        'sparse_intermediate_bounds', False)
    verbosity = self.bound_opts['verbosity']

    assert bound_lower != bound_upper, (
        'we can only optimize lower OR upper bound at one time')
    assert alpha or beta, (
        'nothing to optimize, use compute bound instead!')

    if C is not None:
        self.final_shape = C.size()[:2]
        self.bound_opts.update({'final_shape': self.final_shape})
    if init_alpha:
        # TODO: this should set up aux_reference_bounds.
        self.init_slope(x, share_slopes=opts['use_shared_alpha'],
                   method=method, c=C, final_node_name=final_node_name)

    # Optimizable activations that are actually used and perturbed
    optimizable_activations = [
        n for n in self.optimizable_activations if n.used and n.perturbed]
    # Relu node that are actually used
    relus = [n for n in self.relus if n.used and n.perturbed]

    alphas, betas, parameters = [], [], []
    dense_coeffs_mask = []
    partial_intermediate_layer_bounds = None

    if alpha:
        best_alphas = _set_alpha(
            optimizable_activations, parameters, alphas, lr_alpha)
    
    if topk is not None:
        # TODO remove the hard coded spec size value 
        self.get_cross_execution_params(x, execution_count, parameters, 1.0, topk=topk, spec_size=9)
    elif multiple_execution:
        self.get_cross_execution_params(x, execution_count, parameters, 1.0)
        
    if beta:
        ret_set_beta = _set_beta(
            self, relus, optimizable_activations, single_node_split,
            enable_opt_interm_bounds, betas, opt_coeffs, parameters,
            lr_coeffs, opt_bias, lr_beta, lr_cut_beta, cutter,
            dense_coeffs_mask)
        betas, best_betas, coeffs = ret_set_beta[:3]
        dense_coeffs_mask, best_coeffs, biases, best_biases = ret_set_beta[3:]

    start_time = time()
    if (decision_thresh is not None
            and isinstance(decision_thresh, torch.Tensor)):
        if decision_thresh.dim() == 1:
            # add the spec dim to be aligned with compute_bounds return
            decision_thresh = decision_thresh.unsqueeze(-1)



    target_indices = None
    target_lb = None
    if target_lAs is not None and target_biases is not None:
        target_lb = ptb.concretize(x=None, A=target_lAs) + target_biases
        target_indices = (target_lb <= 0)

    if opt_choice == 'adam-autolr':
        opt = AdamElementLR(parameters)
    elif opt_choice == 'adam':
        opt = optim.Adam(parameters)
    elif opt_choice == 'sgd':
        opt = optim.SGD(parameters, momentum=0.9)
    else:
        raise NotImplementedError(opt_choice)

    # Create a weight vector to scale learning rate.
    loss_weight = torch.ones(size=(x[0].size(0),), device=x[0].device)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, lr_decay)

    if verbosity > 0 and intermediate_beta_enabled:
        self.print_optimized_beta(relus, intermediate_beta_enabled=True)

    # best_intermediate_bounds is linked to aux_reference_bounds!
    best_intermediate_bounds = {}
    if (sparse_intermediate_bounds and aux_reference_bounds is None
            and reference_bounds is not None):
        aux_reference_bounds = {}
        for name, (lb, ub) in reference_bounds.items():
            aux_reference_bounds[name] = [
                lb.detach().clone(), ub.detach().clone()]
    if aux_reference_bounds is None:
        aux_reference_bounds = {}

    with torch.no_grad():
        pruning_in_iteration = False
        # for computing the positive domain ratio
        original_size = x[0].shape[0]
        preserve_mask = None

    # record the overhead due to extra operations from pruning-in-iteration
    pruning_time = 0.

    # print(f'Fixing intermediate bounds {fix_intermediate_layer_bounds}')
    need_grad = True
    patience = 0

    # Before starting optimization clear any stale bounds.
    self._clear_stale_bounds()

    best_target_loss = None
    
    for i in range(iteration):
        if cutter:
            # cuts may be optimized by cutter
            self.cut_module = cutter.cut_module

        intermediate_constr = None
        if i > 0:
            self.track_bounds = True 

        if not fix_intermediate_layer_bounds:
            # If we still optimize all intermediate neurons, we can use
            # intermediate_layer_bounds as reference bounds.
            reference_bounds = intermediate_layer_bounds

        if i == iteration - 1:
            # No grad update needed for the last iteration
            need_grad = False

            if (self.device == 'cuda'
                    and torch.get_default_dtype() == torch.float32
                    and use_float64_in_last_iteration):
                C, x, intermediate_layer_bounds = _to_float64(
                    self, C, x, aux_reference_bounds, intermediate_layer_bounds)

        # we will use last update preserve mask in caller functions to recover
        # lA, l, u, etc to full batch size
        self.last_update_preserve_mask = preserve_mask
        with torch.no_grad() if not need_grad else ExitStack():
            # ret is lb, ub or lb, ub, A_dict (if return_A is set to true)

            # argument for intermediate_layer_bounds
            # If we set neuron bounds individually, or if we are optimizing
            # intermediate layer bounds using beta, we do not set
            # intermediate_layer_bounds.
            # When intermediate betas are used, we must set
            # intermediate_layer_bounds to None because we want to recompute
            # all intermediate layer bounds.
            if beta and intermediate_beta_enabled:
                arg_ilb = partial_intermediate_layer_bounds
            elif fix_intermediate_layer_bounds:
                arg_ilb = intermediate_layer_bounds
            else:
                arg_ilb = None

            # argument for aux_reference_bounds
            if sparse_intermediate_bounds:
                arg_arb = aux_reference_bounds
            else:
                arg_arb = None
            


            ret = self.compute_bounds(
                x, aux, C, method=method, IBP=IBP, forward=forward,
                bound_lower=bound_lower, bound_upper=bound_upper,
                reuse_ibp=reuse_ibp, return_A=return_A, reuse_alpha=True,
                final_node_name=final_node_name, average_A=average_A,
                intermediate_layer_bounds=arg_ilb,
                # This is the currently tightest interval, which will be used to
                # pass split constraints when intermediate betas are used.
                reference_bounds=reference_bounds,
                # This is the interval used for checking for unstable neurons.
                aux_reference_bounds=arg_arb,
                # These are intermediate layer beta variables and their
                # corresponding A matrices and biases.
                intermediate_constr=intermediate_constr,
                needed_A_dict=needed_A_dict,
                update_mask=preserve_mask,
                multiple_execution=multiple_execution, execution_count=execution_count, 
                ptb=ptb, unperturbed_images=unperturbed_images, 
                baseline_refined_bound=baseline_refined_bound,
                intermediate_bound_refinement=intermediate_bound_refinement,
                always_correct_cross_execution=always_correct_cross_execution, 
                cross_refinement_results=cross_refinement_results,
                populate_trace=populate_trace, target_lAs=target_lAs, target_biases=target_biases,
                parent_loss=parent_loss, topk=topk)


        ret_l, ret_u = ret[0], ret[1]

        # print(f'final node name {self.final_name} input node name {self.input_node_name}')
        if target_lAs is not None:
            lb_coef = ret[2]["final_coef"]['Final_Backprop_ANS']['lA']
            lb_bias = ret[2]["final_coef"]['Final_Backprop_ANS']['lbias']
            targeted_loss = self.get_targeted_loss(lb_coef=lb_coef, lb_bias=lb_bias, target_lAs=target_lAs, target_biases=target_biases, ptb=ptb)
            if best_target_loss is None:
                best_target_loss = targeted_loss.detach()
            else:
                best_target_loss = torch.max(targeted_loss.detach(), best_target_loss)
            
            # import pdb; pdb.set_trace()
            if target_indices is not None:
                targeted_loss = targeted_loss[target_indices]
            aggregated_targetted_loss = -targeted_loss.sum()
        elif topk is not None:
            lb_coef = ret[2]["final_coef"]['Final_Backprop_ANS']['lA']
            lb_bias = ret[2]["final_coef"]['Final_Backprop_ANS']['lbias']
            topk_loss = self.get_topk_loss(lb_coef=lb_coef, lb_bias=lb_bias,
                        lambda_t=self.optimizable_lambda, lb=ret_l, ptb=ptb)
            aggregated_topk_loss = -topk_loss.sum()
        elif len(ret) > 2 and multiple_execution: 
            lb_coef = ret[2]["final_coef"]['Final_Backprop_ANS']['lA']
            lb_bias = ret[2]["final_coef"]['Final_Backprop_ANS']['lbias']
            approx_cross_execution_loss = self.get_cross_execution_loss(lb_coef, lb_bias, execution_count, ptb, unperturbed_images)
            final_cross_executional_loss = None
            if always_correct_cross_execution:
                final_cross_executional_loss = self.get_final_cross_executional_loss(lb_coef, lb_bias, execution_count, ret_l,
                                        ptb, unperturbed_images, approx_cross_execution_loss)
                cross_execution_loss = -final_cross_executional_loss.sum()
                # print(f'cross executional loss {final_cross_executional_loss}')
            else:
                cross_execution_loss = -approx_cross_execution_loss.sum()
            with torch.no_grad():            
                if populate_trace:
                    if 'cross_refinement_trace' not in cross_refinement_results.keys():
                        cross_refinement_results['cross_refinement_trace'] = []
                        cross_refinement_results['cross_refinement_time_trace'] = []
                    if not always_correct_cross_execution:
                        raise ValueError(f'Only approximate cross executional loss is avaiable')
                    cross_refinement_results['cross_refinement_trace'].append(final_cross_executional_loss.detach().clone())
                    cross_refinement_results['cross_refinement_time_trace'].append(time() - start_time)
                # print(f'stored cross ex bound {final_cross_executional_loss}')  
            
            if i == iteration - 1:
                print(f'approx cross execution loss {approx_cross_execution_loss}')
        else:
            cross_execution_loss = None
        # import pdb; pdb.set_trace()

        with torch.no_grad():       
            if not multiple_execution and populate_trace:
                if 'base_refinement_trace' not in cross_refinement_results.keys():
                    cross_refinement_results['base_refinement_trace'] = []
                    cross_refinement_results['base_refinement_time_trace'] = []
                cross_refinement_results['base_refinement_trace'].append(ret_l.detach().clone().min(axis=1)[0])
                cross_refinement_results['base_refinement_time_trace'].append(time() - start_time)
                # print(f'stored baseline trace {ret_l.detach().clone().min(axis=1)[0]}')

        if (self.cut_used and i % cutter.log_interval == 0
                and len(self.cut_beta_params) > 0):
            # betas[-1]: (2(0 lower, 1 upper), spec, batch, num_constrs)
            if ret_l is not None:
                print(
                    i, 'lb beta sum:',
                    f'{self.cut_beta_params[-1][0].sum() / ret_l.size(0)},',
                    f'worst {ret_l.min()}')
            if ret_u is not None:
                print(
                    i, 'lb beta sum:',
                    f'{self.cut_beta_params[-1][1].sum() / ret_u.size(0)},',
                    f'worst {ret_u.min()}')

        if i == 0:
            # save results at the first iteration
            best_ret = []
            best_ret_l = _save_ret_first_time(
                ret[0], float('-inf'), x, best_ret)
            best_ret_u = _save_ret_first_time(
                ret[1], float('inf'), x, best_ret)
            ret_0 = ret[0].detach().clone() if bound_lower else ret[1].detach().clone()

            for node in optimizable_activations:
                new_intermediate = [
                    node.inputs[0].lower.detach().clone(),
                    node.inputs[0].upper.detach().clone()]
                best_intermediate_bounds[node.name] = new_intermediate
                if sparse_intermediate_bounds:
                    # Always using the best bounds so far as the reference
                    # bounds.
                    aux_reference_bounds[node.inputs[0].name] = new_intermediate

        l = ret_l
            
        # Reduction over the spec dimension.
        if ret_l is not None and ret_l.shape[1] != 1:
            l = loss_reduction_func(ret_l)
        u = ret_u
        if ret_u is not None and ret_u.shape[1] != 1:
            u = loss_reduction_func(ret_u)

        # full_l, full_ret_l and full_u, full_ret_u is used for update the best
        full_ret_l, full_ret_u = ret_l, ret_u
        full_l = l
        full_ret = ret

        # positive domains may already be filtered out, so we use all domains -
        # negative domains to compute
        if decision_thresh is not None:
            if (isinstance(decision_thresh, torch.Tensor)
                    and decision_thresh.numel() > 1
                    and preserve_mask is not None):
                if decision_thresh.shape[-1] == 1:
                    # single spec with pruned domains
                    try:
                        negative_domain = (
                            ret_l.view(-1)
                            <= decision_thresh[preserve_mask].view(-1)).sum()
                    except:
                        import pdb;pdb.set_trace()
                else:
                    # multiple spec with pruned domains
                    negative_domain = multi_spec_keep_func(
                        ret_l <= decision_thresh[preserve_mask]).sum()
            else:
                if ret_l.shape[-1] == 1:
                    # single spec
                    negative_domain = (
                        ret_l.view(-1) <= decision_thresh.view(-1)).sum()
                else:
                    # multiple spec
                    negative_domain = multi_spec_keep_func(
                        ret_l <= decision_thresh).sum()
            positive_domain_num = original_size - negative_domain
        else:
            positive_domain_num = -1
        positive_domain_ratio = float(
            positive_domain_num) / float(original_size)
        # threshold is 10% by default
        next_iter_pruning_in_iteration = (
            opts['pruning_in_iteration'] and decision_thresh is not None
            and positive_domain_ratio > opts['pruning_in_iteration_threshold'])

        if pruning_in_iteration:
            stime = time()
            if return_A:
                raise Exception(
                    'Pruning in iteration optimization does not support '
                    'return A yet. '
                    'Please fix or discard this optimization by setting '
                    '--disable_pruning_in_iteration '
                    'or bab: pruning_in_iteration: false')
            now_preserve_mask = _get_preserve_mask(
                decision_thresh, ret_l, preserve_mask, multi_spec_keep_func)
            # prune C
            if C is not None and C.shape[0] == x[0].shape[0]:
                C = C[now_preserve_mask]  # means C is also batch specific
            # prune x
            x, pre_prune_size = _prune_x(x, now_preserve_mask)
            # prune bounds
            ret_prune = _prune_bounds_by_mask(
                now_preserve_mask, decision_thresh, ret_l, ret_u, ret,
                preserve_mask, epsilon_over_decision_thresh, original_size,
                loss_reduction_func, beta, intermediate_beta_enabled,
                fix_intermediate_layer_bounds,
                intermediate_layer_bounds, aux_reference_bounds,
                partial_intermediate_layer_bounds, pre_prune_size)
            full_l, full_ret_l = ret_prune[:2]
            # ret_prune[2] is full_u which is unused
            full_ret_u, full_ret, preserve_mask_next = ret_prune[3:]
            pruning_time += time() - stime

        loss_ = l if bound_lower else -u
        # print(f'iteration {i} lower bound: {loss_}')
        stop_criterion = stop_criterion_func(
            full_ret_l) if bound_lower else stop_criterion_func(-full_ret_u)
        if (type(stop_criterion) != bool
                and stop_criterion.numel() > 1 and pruning_in_iteration):
            stop_criterion = stop_criterion[preserve_mask]
        total_loss = -1 * loss_
        if type(stop_criterion) == bool:
            loss = total_loss.sum() * (not stop_criterion)
        else:
            loss = (total_loss * stop_criterion.logical_not()).sum()


        stop_criterion_final = isinstance(
            stop_criterion, torch.Tensor) and stop_criterion.all()

        if i == iteration - 1:
            best_ret = list(best_ret)
            if best_ret[0] is not None:
                best_ret[0] = best_ret[0].to(torch.get_default_dtype())
            if best_ret[1] is not None:
                best_ret[1] = best_ret[1].to(torch.get_default_dtype())

        if (i == iteration - 1 and self.device == 'cuda'
                and torch.get_default_dtype() == torch.float32
                and use_float64_in_last_iteration):
            total_loss, x, full_ret = _to_default_dtype(
                self, x, total_loss, full_ret, ret, best_intermediate_bounds,
                return_A)

        with torch.no_grad():
            # for lb and ub, we update them in every iteration since updating
            # them is cheap
            need_update = False
            if keep_best:
                if best_ret_u is not None:
                    ret_upd = _update_best_ret(
                        full_ret_u, best_ret_u, full_ret, best_ret, need_update,
                        idx=1)
                    best_ret_u, best_ret, _, _, need_update = ret_upd
                if best_ret_l is not None:
                    ret_upd = _update_best_ret(
                        full_ret_l, best_ret_l, full_ret, best_ret, need_update,
                        idx=0)
                    best_ret_l, best_ret, _, _, need_update = ret_upd
            else:
                # Not saving the best, just keep the last iteration.
                if full_ret[0] is not None:
                    best_ret[0] = full_ret[0]
                if full_ret[1] is not None:
                    best_ret[1] = full_ret[1]
            if return_A:
                # FIXME: A should also be updated by idx.
                best_ret = [best_ret[0], best_ret[1], full_ret[2]]

            if need_update:
                patience = 0  # bounds improved, reset patience
            else:
                patience += 1

            # Save variables if this is the best iteration.
            # To save computational cost, we only check keep_best at the first
            # (in case divergence) and second half iterations
            # or before early stop by either stop_criterion or
            # early_stop_patience reached
            if (i < 1 or i > int(iteration * start_save_best)
                    or stop_criterion_final or patience == early_stop_patience):

                # compare with the first iteration results and get improved indexes
                if bound_lower:
                    idx_mask, idx = _get_idx_mask(0, full_ret_l, ret_0)
                    ret_0[idx] = full_ret_l[idx]
                else:
                    idx_mask, idx = _get_idx_mask(1, full_ret_u, ret_0)
                    ret_0[idx] = full_ret_u[idx]

                if idx is not None:
                    # for update propose, we condition the idx to update only
                    # on domains preserved
                    if pruning_in_iteration:
                        # local sparse index of preserved samples where
                        # idx == true
                        local_idx = idx_mask[preserve_mask].nonzero().view(-1)
                        # idx is global sparse index of preserved samples where
                        # idx == true
                        new_idx = torch.zeros_like(
                            idx_mask, dtype=torch.bool, device=x[0].device)
                        new_idx[preserve_mask] = idx_mask[preserve_mask]
                        idx = new_idx.nonzero().view(-1)
                        reference_idx = local_idx
                    else:
                        reference_idx = idx

                    _update_optimizable_activations(
                        optimizable_activations, intermediate_layer_bounds,
                        fix_intermediate_layer_bounds, best_intermediate_bounds,
                        reference_idx, idx, alpha, best_alphas)

                    if beta and single_node_split:
                        _update_best_beta(
                            self, enable_opt_interm_bounds, betas,
                            optimizable_activations, best_betas, idx)


        # if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
            # print(f'****** iter [{i}]',
            #     f'loss: {loss.item()}, lr: {opt.param_groups[0]["lr"]}')
        if stop_criterion_final:
            if multiple_execution:
                print(f'cross_execution_loss {approx_cross_execution_loss}')
            print(f'\nall verified at {i}th iter')
            break

        if patience > early_stop_patience:
            if multiple_execution:
                print(f'cross_execution_loss {approx_cross_execution_loss}')
            print(
                f'Early stop at {i}th iter due to {early_stop_patience}'
                ' iterations no improvement!')
            break

        current_lr = [param_group['lr'] for param_group in opt.param_groups]

        opt.zero_grad(set_to_none=True)

        if verbosity > 2:
            print(
                f'*** iter [{i}]\n', f'loss: {loss.item()}',
                total_loss.squeeze().detach().cpu().numpy(), 'lr: ',
                current_lr)
            if beta:
                self.print_optimized_beta(relus, intermediate_beta_enabled)
                if opt_coeffs:
                    for co in coeffs:
                        print(f'coeff sum: {co.abs().sum():.5g}')
            if beta and i == 0 and verbosity > 2:
                breakpoint()

        if i != iteration - 1:
            # we do not need to update parameters in the last step since the
            # best result already obtained
            retain_graph = False
            if target_lAs is not None:
                aggregated_targetted_loss.backward(retain_graph=retain_graph)
            elif topk is not None:
                aggregated_topk_loss.backward(retain_graph=retain_graph)
            elif multiple_execution:
                cross_execution_loss.backward(retain_graph=retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)
            # if multiple_execution:
            #     cross_execution_loss.backward()
            # else:
            #     loss.backward()
            # All intermediate variables are not needed at this point.
            self._clear_and_set_new(None)
            if opt_choice == 'adam-autolr':
                opt.step(lr_scale=[loss_weight, loss_weight])
            else:
                opt.step()

        if beta:
            # Clipping to >=0.
            for b in betas:
                b.data = (b >= 0) * b.data
            for dmi in range(len(dense_coeffs_mask)):
                # apply dense mask to the dense split coeffs matrix
                coeffs[dmi].data = (
                    dense_coeffs_mask[dmi].float() * coeffs[dmi].data)


        if alpha:
            for m in optimizable_activations:
                m.clip_alpha_()

        scheduler.step()

        if pruning_in_iteration:
            preserve_mask = preserve_mask_next
        if not pruning_in_iteration and next_iter_pruning_in_iteration:
            # init preserve_mask etc
            preserve_mask = torch.arange(
                0, x[0].shape[0], device=x[0].device, dtype=torch.long)
            pruning_in_iteration = True

    self.detach_bounds()

    # correct the final targeted_loss
    if target_lAs is not None:
        lb_coef = ret[2]["final_coef"]['Final_Backprop_ANS']['lA']
        lb_bias = ret[2]["final_coef"]['Final_Backprop_ANS']['lbias']
        targetted_loss = self.get_targeted_loss(lb_coef=lb_coef, lb_bias=lb_bias, target_lAs=target_lAs, target_biases=target_biases, ptb=ptb)
        targetted_loss = torch.max(best_target_loss, targetted_loss.detach())
        if parent_loss is None:
            parent_loss = torch.zeros_like(targetted_loss, 
                                    device=targetted_loss.device)
            parent_loss[0, :] = -1e9
        targetted_loss = torch.max(targetted_loss, parent_loss)
        cross_refinement_results['targetted_loss'] = targetted_loss.detach().clone()
    # correct the final cross executional loss.
    elif topk is not None:
        lb_coef = ret[2]["final_coef"]['Final_Backprop_ANS']['lA']
        lb_bias = ret[2]["final_coef"]['Final_Backprop_ANS']['lbias']
        final_topk_loss = self.get_final_topk_loss(lb_coef=lb_coef, lb_bias=lb_bias, 
                                                  lambda_t=self.optimizable_lambda, 
                                                  lb=ret_l, ptb=ptb, topk=topk)
        if parent_loss is None:
            parent_loss = torch.zeros_like(final_topk_loss, device = final_topk_loss.device) - 1e9
        
        assert parent_loss.shape == final_topk_loss.shape
        
        final_topk_loss = torch.max(final_topk_loss, parent_loss)
        cross_refinement_results['final_topk_loss'] = final_topk_loss.detach().clone()  
    elif len(ret) > 2 and multiple_execution: 
        lb_coef = ret[2]["final_coef"]['Final_Backprop_ANS']['lA']
        lb_bias = ret[2]["final_coef"]['Final_Backprop_ANS']['lbias']
        final_cross_executional_loss = self.get_final_cross_executional_loss(lb_coef=lb_coef, lb_bias=lb_bias, execution_count=execution_count,
                                            lower_bound=ret_l.detach(), approx_cross_ex_loss_tensor=approx_cross_execution_loss,
                                            ptb=ptb, unperturbed_images=unperturbed_images)
        if best_ret is not None:
            best_ret_min = best_ret[0].min(dim=1).values
            best_ret_min = best_ret_min.view(execution_count, -1)
            best_ret_min = best_ret_min.max(dim=0).values
            final_cross_executional_loss = torch.max(final_cross_executional_loss.detach().clone(), best_ret_min)

        if parent_loss is None:
            parent_loss = torch.zeros_like(final_cross_executional_loss, device = final_cross_executional_loss.device) - 1e9
        
        assert parent_loss.shape == final_cross_executional_loss.shape
        
        final_cross_executional_loss = torch.max(final_cross_executional_loss, parent_loss)
        cross_refinement_results['final_loss'] = final_cross_executional_loss.detach().clone()
        print(f'Final cross executional loss {final_cross_executional_loss}')  
    
    if pruning_in_iteration:
        # overwrite pruned cells in best_ret by threshold + eps
        if return_A:
            fin_l, fin_u, fin_A = best_ret
        else:
            fin_l, fin_u = best_ret
            fin_A = None
        if fin_l is not None:
            new_fin_l = full_ret_l
            new_fin_l[preserve_mask] = fin_l[preserve_mask]
            fin_l = new_fin_l
        if fin_u is not None:
            new_fin_u = full_ret_u
            new_fin_u[preserve_mask] = fin_u[preserve_mask]
            fin_u = new_fin_u
        if return_A:
            best_ret = (fin_l, fin_u, fin_A)
        else:
            best_ret = (fin_l, fin_u)

    if verbosity > 3:
        breakpoint()

    if keep_best:
        def update_best(dest, src):
            for item_dest, item_src in zip(dest, src):
                if enable_opt_interm_bounds:
                    for key in item_dest.keys():
                        item_dest[key].data = item_src[key].data
                else:
                    item_dest.data = item_src.data

        # Set all variables to their saved best values.
        with torch.no_grad():
            for idx, node in enumerate(optimizable_activations):
                if alpha:
                    # Assigns a new dictionary.
                    node.alpha = best_alphas[node.name]
                # Update best intermediate layer bounds only when they are
                # optimized. If they are already fixed in
                # intermediate_layer_bounds, then do nothing.
                best_intermediate = best_intermediate_bounds[node.name]
                node.inputs[0].lower.data = best_intermediate[0].data
                node.inputs[0].upper.data = best_intermediate[1].data
                if beta:
                    if (single_node_split and hasattr(node, 'sparse_beta')
                            and node.sparse_beta is not None):
                        if enable_opt_interm_bounds:
                            for key in node.sparse_beta.keys():
                                node.sparse_beta[key].copy_(
                                    best_betas[node.name][key])
                        else:
                            node.sparse_beta.copy_(best_betas[idx])
                    else:
                        update_best(betas, best_betas)
                        if opt_coeffs:
                            update_best(coeffs, best_coeffs)
                        if opt_bias:
                            update_best(biases, best_biases)
            if self.cut_used:
                regular_beta_length = len(betas) - len(self.cut_beta_params)
                for ii in range(len(self.cut_beta_params)):
                    self.cut_beta_params[ii].data = best_betas[
                        regular_beta_length + ii].data

    if (intermediate_layer_bounds is not None
            and not fix_intermediate_layer_bounds):
        for l in self._modules.values():
            if (l.name in intermediate_layer_bounds.keys()
                    and hasattr(l, 'lower')):
                l.lower = torch.max(
                    l.lower, intermediate_layer_bounds[l.name][0])
                l.upper = torch.min(
                    l.upper, intermediate_layer_bounds[l.name][1])
                infeasible_neurons = l.lower > l.upper
                if infeasible_neurons.any():
                    print(
                        f'Infeasibility detected in layer {l.name}.',
                        infeasible_neurons.sum().item(),
                        infeasible_neurons.nonzero()[:, 0])

    if verbosity > 0:
        if self.cut_used and beta:
            print(
                'first 10 best general betas:',
                best_betas[-1].view(2, -1)[0][:10], 'sum:',
                best_betas[-1][0].sum().item())
        if best_ret_l is not None:
            # FIXME: unify the handling of l and u.
            print(
                'best_l after optimization:',
                best_ret_l.sum().item(), 'with beta sum per layer:',
                [p.sum().item() for p in betas])
        # print('alpha/beta optimization time:', time() - start)
    
    for node in optimizable_activations:
        node.opt_end()

    # update pruning ratio
    if (opts['pruning_in_iteration'] and decision_thresh is not None
            and full_l.numel() > 0):
        stime = time()
        with torch.no_grad():
            if isinstance(decision_thresh, torch.Tensor):
                if decision_thresh.shape[-1] == 1:
                    neg_domain_num = torch.sum(
                        full_ret_l.view(-1) <= decision_thresh.view(-1)).item()
                else:
                    neg_domain_num = torch.sum(multi_spec_keep_func(
                        full_ret_l <= decision_thresh)).item()
            else:
                if full_l.shape[-1] == 1:
                    neg_domain_num = torch.sum(
                        full_ret_l.view(-1) <= decision_thresh).item()
                else:
                    neg_domain_num = torch.sum(multi_spec_keep_func(
                        full_ret_l <= decision_thresh)).item()
            now_pruning_ratio = (1.0 -
                float(neg_domain_num) / float(full_l.shape[0]))
            print('pruning_in_iteration open status:', pruning_in_iteration)
            print(
                'ratio of positive domain =', full_l.shape[0] - neg_domain_num,
                '/', full_l.numel(), '=', now_pruning_ratio)
        pruning_time += time() - stime
        print('pruning-in-iteration extra time:', pruning_time)
    
    self.track_bounds = False
    return best_ret


def init_slope(
        self, x, share_slopes=False, method='backward',
        c=None, bound_lower=True, bound_upper=True, final_node_name=None,
        intermediate_layer_bounds=None, activation_opt_params=None,
        skip_bound_compute=False, return_A=False, needed_A_dict=None, multiple_execution=False, execution_count=1, 
        ptb=None, unperturbed_images=None, iteration = None, baseline_refined_bound={},
        intermediate_bound_refinement=False, 
        always_correct_cross_execution=False,
        cross_refinement_results={}, populate_trace=False, target_lAs=None,
        target_biases=None, parent_loss=None, topk=None):
    for node in self.optimizable_activations:
        # initialize the parameters
        node.opt_init()

    if (not skip_bound_compute or intermediate_layer_bounds is None or
            activation_opt_params is None or not all(
                [relu.name in activation_opt_params for relu in self.relus])):
        skipped = False
        # if new interval is None, then CROWN interval is not present
        # in this case, we still need to redo a CROWN pass to initialize
        # lower/upper
        with torch.no_grad():
            ret = self.compute_bounds(
                x=x, C=c, method=method, bound_lower=bound_lower,
                bound_upper=bound_upper, final_node_name=final_node_name,
                intermediate_layer_bounds=intermediate_layer_bounds, return_A= return_A, needed_A_dict= needed_A_dict, multiple_execution=multiple_execution, execution_count=execution_count, 
                ptb=ptb, unperturbed_images=unperturbed_images, iteration = iteration,
                baseline_refined_bound=baseline_refined_bound,
                intermediate_bound_refinement=intermediate_bound_refinement,
                always_correct_cross_execution=always_correct_cross_execution, 
                cross_refinement_results=cross_refinement_results,
                populate_trace=populate_trace, target_lAs=target_lAs, 
                target_biases=target_biases, parent_loss=parent_loss, topk=topk)
            if return_A:
                l, u, A = ret
            else:
                l, u = ret
    else:
        # we skip, but we still would like to figure out the "used",
        # "perturbed", "backward_from" of each note in the graph
        skipped = True
        # this set the "perturbed" property
        self._set_input(
            *x, intermediate_layer_bounds=intermediate_layer_bounds)
        self._clear_stale_bounds()

        final = self.final_node(
        ) if final_node_name is None else self[final_node_name]
        self._set_used_nodes(final)

        self.backward_from = {node: [final] for node in self._modules}

    final_node_name = final_node_name or self.final_name

    init_intermediate_bounds = {}
    for node in self.optimizable_activations:
        if not node.used or not node.perturbed:
            continue
        start_nodes = []
        if method in ['forward', 'forward+backward']:
            start_nodes.append(('_forward', 1, None))
        if method in ['backward', 'forward+backward']:
            start_nodes += self.get_alpha_crown_start_nodes(
                node, c=c, share_slopes=share_slopes,
                final_node_name=final_node_name)
        if skipped:
            node.restore_optimized_params(activation_opt_params[node.name])
        else:
            node.init_opt_parameters(start_nodes)
        init_intermediate_bounds[node.inputs[0].name] = (
            [node.inputs[0].lower.detach(), node.inputs[0].upper.detach()])

    if self.bound_opts['verbosity'] >= 1:
        print('Optimizable variables initialized.')
    if skip_bound_compute:
        return init_intermediate_bounds
    else:
        if return_A:
            return l, u, init_intermediate_bounds, A
        else:
            return l, u, init_intermediate_bounds
