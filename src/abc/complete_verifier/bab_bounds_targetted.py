import copy
import time
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import numpy as np
import arguments
import warnings
import multiprocessing

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import stop_criterion_sum

from lp_mip_solver import *
from attack_pgd import attack
from cuts import Cutter
import sys
import os

total_func_time = total_prepare_time = total_bound_time = total_beta_bound_time = total_transfer_time = total_finalize_time = 0.0


def reset_targetted_beta(self, batch, max_splits_per_layer=None, all_img_betas=None):
    # Recreate new beta with appropriate shape.
    for mi, m in enumerate(self.net.perturbed_optimizable_activations):
        if isinstance(m, BoundRelu):
            # Create only the non-zero beta. For each layer, it is padded to maximal length.
            # We create tensors on CPU first, and they will be transferred to GPU after initialized.
            if arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']:
                raise NotImplementedError("not implemented")
            else:
                m.sparse_beta_loc = torch.zeros(size= (batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
                m.sparse_beta_sign = torch.zeros(size= (batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                
                sparse_betas = []
                for betas in all_img_betas:
                    sparse_beta = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                    
                    # Load beta from history.
                    for bi in range(batch):
                        if betas is not None and betas[bi] is not None:
                            # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                            try:
                                valid_betas = len(betas[bi][mi])
                                sparse_beta[bi, :valid_betas] = betas[bi][mi]
                            except:
                                import pdb;pdb.set_trace()
                                raise RuntimeError
                    # This is the beta variable to be optimized for this layer.
                    sparse_betas.append(sparse_beta.repeat(2, 1))
                m.sparse_beta = torch.cat(sparse_betas).detach().to(device=self.net.device, non_blocking=True).requires_grad_()
                

def set_beta_sign_loc(self, batch, num_images, decision, history):
    for bi in range(batch):
        # Add history splits.
        d, idx = decision[bi][0], decision[bi][1]
        # Each history element has format [[[layer 1's split location], [layer 1's split coefficients +1/-1]], [[layer 2's split location], [layer 2's split coefficients +1/-1]], ...].
        for mi, (split_locs, split_coeffs) in enumerate(history[bi]):
            split_len = len(split_locs)
            self.net.relus[mi].sparse_beta_sign[bi, :split_len] = torch.as_tensor(split_coeffs, device='cpu', dtype=torch.get_default_dtype())
            self.net.relus[mi].sparse_beta_loc[bi, :split_len] = torch.as_tensor(split_locs, device='cpu', dtype=torch.int64)
            # Add current decision for positive splits.
            if mi == d:
                self.net.relus[mi].sparse_beta_sign[bi, split_len] = 1.0
                self.net.relus[mi].sparse_beta_loc[bi, split_len] = idx
    
    # Duplicate split location.
    for m in self.net.relus:
        m.sparse_beta_loc = m.sparse_beta_loc.repeat(2, 1).detach()
        m.sparse_beta_loc = m.sparse_beta_loc.to(device=self.net.device, non_blocking=True)
        m.sparse_beta_sign = m.sparse_beta_sign.repeat(2, 1).detach()
    # Fixup the second half of the split (negative splits).
    for bi in range(batch):
        d = decision[bi][0]  # layer of this split.
        split_len = len(history[bi][d][0])  # length of history splits for this example in this layer.
        self.net.relus[d].sparse_beta_sign[bi + batch, split_len] = -1.0
    # Transfer tensors to GPU.
    for m in self.net.relus:
        m.sparse_beta_sign = m.sparse_beta_sign.to(device=self.net.device, non_blocking=True)
    #repeat each beta sign and loc num_image times
    for m in self.net.relus:
        m.sparse_beta_sign = m.sparse_beta_sign.repeat(num_images, 1).detach()
        m.sparse_beta_loc = m.sparse_beta_loc.repeat(num_images, 1).detach()
        

def get_beta_multiple(model):  
    retb = []
    for mi, m in enumerate(model.perturbed_optimizable_activations):
            if hasattr(m, 'sparse_beta'):
                retb.append(m.sparse_beta)

    return retb 
    
    

def expander(tensor, batch):
    with torch.no_grad():
        return tensor.repeat(1, batch * 2, *[1] * (tensor.ndim - 2)).reshape((tensor.shape[0] * batch * 2, *tensor.shape[1:]))

def update_bounds_parallel_targetted(
            self, split=None, history = None, fix_intermediate_layer_bounds=True, shortcut=False,
            decision_thresh=None, stop_criterion_func=stop_criterion_sum(0),
            multi_spec_keep_func=None, all_img_pre_lb = None, all_img_pre_ub = None, all_img_betas = None, all_img_slopes = None, all_img_cs = None, target_lAs = None, target_biases = None, all_img_targetted_loss = None):
        global total_func_time, total_bound_time, total_prepare_time, total_beta_bound_time, total_transfer_time, total_finalize_time
        
        
        beta = arguments.Config["solver"]["beta-crown"]["beta"] # might need to set beta False in FSB node selection
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        iteration = arguments.Config["solver"]["beta-crown"]["iteration"]
        lr_alpha = arguments.Config["solver"]["beta-crown"]['lr_alpha']
        lr_beta = arguments.Config["solver"]["beta-crown"]["lr_beta"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
        enable_opt_interm_bounds = arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']
        pruning_in_iteration = arguments.Config["bab"]["pruning_in_iteration"]
        pruning_in_iteration_threshold = arguments.Config["bab"]["pruning_in_iteration_ratio"]
        cut_iteration = arguments.Config["bab"]["cut"]["bab_iteration"]
        lr_cut_beta = arguments.Config["bab"]["cut"]["lr_beta"]
        cut_lr = arguments.Config["bab"]["cut"]["lr"]
        num_images = len(all_img_pre_lb)
        func_time = time.time()
        prepare_time = bound_time = transfer_time = finalize_time = beta_bound_time = 0.0
        diving_batch = 0
        if type(split) == list:
            decision = np.array(split)
        else:
            decision = np.array(split["decision"])
            decision = np.array([i.squeeze() for i in decision])

        batch = len(decision)

        if "diving" in split:
            diving_batch = split["diving"]
            print(f"regular batch size: 2*{batch}, diving batch size 1*{diving_batch}")
        
        assert diving_batch == 0, "Diving Batch is Not Implemented for Multiple Image Splitting"

        ret_s = [[] for _ in range(batch * 2 + diving_batch)]
        ret_b = [[] for _ in range(batch * 2 + diving_batch)]
        # Each key is corresponding to a pre-relu layer, and each value intermediate
        # beta values for neurons in that layer.
        new_split_history = [[{} for _ in range(batch * 2)] for _ in range(num_images)]
        best_intermediate_betas = [[defaultdict(dict) for _ in range(batch * 2)] for _ in range(num_images)] # Each key is corresponding to a pre-relu layer, and each value intermediate beta values for neurons in that layer.

        start_prepare_time = time.time()
        # iteratively change upper and lower bound from former to later layer

        self.net.cut_beta_params = []
        if self.net.cut_used:
            # disable cut_used for branching node selection, reenable when beta is True
            print('cut disabled for branching node selection')
            self.net.cut_used = False
            for m in self.net.relus:
                m.cut_used = False
            self.net.cut_beta_params = []
        
        if beta:
            # count how many split nodes in each batch example (batch, num of layers)
            splits_per_example = torch.zeros(
                size=(batch, len(self.net.relus)), dtype=torch.int64, device='cpu',
                requires_grad=False)
            for bi in range(batch):
                d = decision[bi][0]
                for mi, layer_splits in enumerate(history[bi]):
                    splits_per_example[bi, mi] = len(layer_splits[0]) + int(d == mi)  # First element of layer_splits is a list of split neuron IDs.
            # This is the maximum number of split in each relu neuron for each batch.
            if batch > 0: max_splits_per_layer = splits_per_example.max(dim=0)[0]
            


            # Create and load warmup beta.
            self.reset_targetted_beta(batch, max_splits_per_layer=max_splits_per_layer, all_img_betas=all_img_betas)  # warm start beta
            
            if arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']:
                raise NotImplementedError("not implemented")
            else:
                self.set_beta_sign_loc(batch, num_images, decision, history)


            self.net.cut_used = arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["bab_cut"]
            # even we need to use cut, maybe the cut is not fetched yet

            batch_size = batch * 2 + diving_batch
            if self.net.cut_used and getattr(self.net, "cut_module", None) is not None:
                raise NotImplementedError("not implemented")
        else:
            for m in self.net.relus:
                m.beta = None
        
        # pre_ub_all[:-1] means pre-set bounds for all intermediate layers
        with torch.no_grad():
            # # Only the last element is used later.
            pre_lb_last = torch.cat([torch.cat([all_img_pre_lb[im][-1][:batch], all_img_pre_lb[im][-1][:batch], all_img_pre_lb[im][-1][batch:]]) for im in range(len(all_img_pre_lb))])
            pre_ub_last = torch.cat([torch.cat([all_img_pre_ub[im][-1][:batch], all_img_pre_ub[im][-1][:batch],  all_img_pre_ub[im][-1][batch:]]) for im in range(len(all_img_pre_ub))])
            
            double_cs = torch.cat([torch.cat([all_img_cs[im][:batch], all_img_cs[im][:batch], all_img_cs[im][batch:]], dim=0) for im in range(len(all_img_cs))])
            parent_loss = torch.cat([torch.cat([all_img_targetted_loss[im][:batch], all_img_targetted_loss[im][:batch],  all_img_targetted_loss[im][batch:]]) for im in range(len(all_img_targetted_loss))])
        
        new_intermediate_layer_bounds = self.expand_intermediate_layer_bounds(decision, batch, all_img_pre_lb, all_img_pre_ub)
        
        
        if len(all_img_slopes[0]) > 0:
            # set slope here again
            self.set_slope(self.net, all_img_slopes[0], diving_batch=diving_batch)
        
        self.concat_slopes_multiple(all_img_slopes, 0)
 
        
        new_target_lAs = expander(target_lAs, batch)
        new_target_biases = expander(target_biases, batch)      
        ptb = PerturbationLpNorm(norm= self.x.ptb.norm, eps= self.x.ptb.eps, x_L = expander(self.x.ptb.x_L, batch), x_U = expander(self.x.ptb.x_U, batch))
        
        new_x = BoundedTensor(expander(self.x.data, batch), ptb)
        
        if decision_thresh is not None and isinstance(decision_thresh, torch.Tensor) and decision_thresh.numel() > 1:
            decision_thresh = torch.cat([decision_thresh] * (new_x.shape[0]//len(decision_thresh)), dim=0)
        

        prepare_time += time.time() - start_prepare_time
        start_bound_time = time.time()

        if shortcut:
            self.net.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': beta, 'single_node_split': True,
                                                             'fix_intermediate_layer_bounds': fix_intermediate_layer_bounds,
                                                             'optimizer':optimizer,
                                                             'pruning_in_iteration': pruning_in_iteration,
                                                             'pruning_in_iteration_threshold': pruning_in_iteration_threshold},
                                                            'enable_opt_interm_bounds': enable_opt_interm_bounds,})
            with torch.no_grad():
                lb, _, = self.net.compute_bounds(x=(new_x,), C=double_cs, method='backward', reuse_alpha=True,
                                                 intermediate_layer_bounds=new_intermediate_layer_bounds, bound_upper=False)
            return lb
            

        return_A = True if (get_upper_bound or arguments.Config["cross_ex"]["enable_cross_ex"]) else False  # we need A matrix to construct adv example

        original_size = new_x.shape[0]
        if fix_intermediate_layer_bounds:
            start_beta_bound_time = time.time()
            self.net.set_bound_opts({'optimize_bound_args': {
                'enable_beta_crown': beta, 'single_node_split': True,
                'fix_intermediate_layer_bounds': fix_intermediate_layer_bounds, 'iteration': iteration,
                'lr_alpha': lr_alpha, 'lr_decay': lr_decay, 'lr_beta': lr_beta,
                'optimizer': optimizer,
                'pruning_in_iteration': pruning_in_iteration,
                'pruning_in_iteration_threshold': pruning_in_iteration_threshold,
                'stop_criterion_func': stop_criterion_func,
                'multi_spec_keep_func': multi_spec_keep_func},
                'enable_opt_interm_bounds': enable_opt_interm_bounds,
                'lr_cut_beta': lr_cut_beta,
            })
            kept_layer_names = list(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
            for name in kept_layer_names:
                print(f'Removing intermediate layer bounds for layer {name}.')
                del new_intermediate_layer_bounds[name]
            print(new_x.shape, double_cs.shape, decision_thresh.shape if decision_thresh is not None else None)
            if arguments.Config["cross_ex"]["enable_cross_ex"]:
                tmp_ret = self.net.compute_bounds(
                    x=(new_x,), C= double_cs, method= self.method,
                    intermediate_layer_bounds= new_intermediate_layer_bounds, cutter=self.cutter,
                    bound_upper=False, decision_thresh=decision_thresh, return_A = self.return_A,
                    needed_A_dict = self.needed_A_dict, multiple_execution = self.multiple_execution, execution_count = self.execution_count, ptb = ptb, unperturbed_images = expander(self.unperturbed_images, batch), 
                    iteration = self.iteration, baseline_refined_bound = self.baseline_refined_bound, intermediate_bound_refinement = self.intermediate_bound_refinement,
                    always_correct_cross_execution = self.always_correct_cross_execution, cross_refinement_results = self.cross_refinement_results, 
                    populate_trace = self.populate_trace, target_lAs= new_target_lAs, target_biases= new_target_biases,
                    parent_loss=parent_loss)
            else:
                tmp_ret = self.net.compute_bounds(
                x=(new_x,), C=c, method='CROWN-Optimized',
                intermediate_layer_bounds=new_intermediate_layer_bounds,
                return_A=return_A, needed_A_dict=self.needed_A_dict, cutter=self.cutter,
                bound_upper=False, decision_thresh=decision_thresh, target_lAs=target_lAs)
            beta_bound_time += time.time() - start_beta_bound_time
            # we don't care about the upper bound of the last layer
        else:
            # all intermediate bounds are re-calculated by optimized CROWN
            self.net.set_bound_opts({'optimize_bound_args': {
                'enable_beta_crown': beta, 'fix_intermediate_layer_bounds': fix_intermediate_layer_bounds,
                'iteration': iteration, 'lr_alpha': lr_alpha, 'lr_decay': lr_decay,
                'lr_beta': lr_beta, 'optimizer': optimizer,
                'pruning_in_iteration': pruning_in_iteration,
                'pruning_in_iteration_threshold': pruning_in_iteration_threshold,
                'stop_criterion_func': stop_criterion_func,
                'multi_spec_keep_func': multi_spec_keep_func},
                'enable_opt_interm_bounds': enable_opt_interm_bounds,
                'lr_cut_beta': lr_cut_beta,
            })
            if arguments.Config["cross_ex"]["enable_cross_ex"]:
                tmp_ret = self.net.compute_bounds(
                    x=(new_x,), C=double_cs, method= self.method, intermediate_layer_bounds=new_intermediate_layer_bounds,
                    cutter=self.cutter,bound_upper=False, decision_thresh=decision_thresh, return_A = self.return_A,
                    needed_A_dict = self.needed_A_dict, multiple_execution = self.multiple_execution, execution_count = self.execution_count, ptb = ptb, unperturbed_images = self.unperturbed_images, 
                    iteration = self.iteration, baseline_refined_bound = self.baseline_refined_bound, intermediate_bound_refinement = self.intermediate_bound_refinement,
                    always_correct_cross_execution = self.always_correct_cross_execution, cross_refinement_results = self.cross_refinement_results, populate_trace = self.populate_trace)
            else:
                tmp_ret = self.net.compute_bounds(
                x=(new_x,), C=double_cs, method='CROWN-Optimized', intermediate_layer_bounds=new_intermediate_layer_bounds,
                return_A=return_A, needed_A_dict=self.needed_A_dict, cutter=self.cutter,
                bound_upper=False, decision_thresh=decision_thresh)

        if get_upper_bound:
            lb, _, A = tmp_ret
            primal_x, ub = self.get_primal_upper_bound(A)
        elif arguments.Config["cross_ex"]["enable_cross_ex"]:
            lb, _, A = tmp_ret
            ub = torch.zeros_like(lb) + np.inf # dummy upper bound
            primal_x = None
        else:
            lb, _ = tmp_ret
            ub = torch.zeros_like(lb) + np.inf # dummy upper bound
            primal_x = None

        bound_time += time.time() - start_bound_time

        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            start_transfer_time = time.time()
            lb, ub = lb.to(device='cpu'), ub.to(device='cpu')
            # indexing on GPU seems to be faster, so get_lA_parallel() is conducted on GPU side then move to CPU
            lAs = self.get_lA_parallel(self.net, self.net.last_update_preserve_mask, original_size, to_cpu=True)
            transfer_net = self.transfer_to_cpu(self.net, non_blocking=False)
            transfer_time = time.time() - start_transfer_time

            start_finalize_time = time.time()

            if len(all_img_slopes[0]) > 0:
                ret_s = self.get_slope(transfer_net)
            
            if beta:
                ret_b = get_beta_multiple(transfer_net)

            # Reorganize tensors.
            lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(transfer_net, lb, ub, batch * 2 * num_images, diving_batch=diving_batch)
            
            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_last.cpu())
            if not get_upper_bound:
                # Do not set to min so the primal is always corresponding to the upper bound.
                upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_last.cpu())
            ret_l, ret_u = lower_bounds_new, upper_bounds_new

            finalize_time = time.time() - start_finalize_time


        func_time = time.time() - func_time
        total_func_time += func_time
        total_bound_time += bound_time
        total_beta_bound_time += beta_bound_time
        total_prepare_time += prepare_time
        total_transfer_time += transfer_time
        total_finalize_time += finalize_time
        print(f'This batch time : update_bounds func: {func_time:.4f}\t prepare: {prepare_time:.4f}\t bound: {bound_time:.4f}\t transfer: {transfer_time:.4f}\t finalize: {finalize_time:.4f}')
        print(f'Accumulated time: update_bounds func: {total_func_time:.4f}\t prepare: {total_prepare_time:.4f}\t bound: {total_bound_time:.4f}\t transfer: {total_transfer_time:.4f}\t finalize: {total_finalize_time:.4f}')

        return seperate_per_img(num_images, splits_per_example, ret_u[-1], ret_l[-1], None, ret_l, ret_u, lAs, ret_s, new_split_history, ret_b, best_intermediate_betas, primal_x, double_cs, A) 
   

def seperate_As_per_batch(A, num_images, batch):
    new_As = [{} for _ in range(batch)]
    for k, v in A.items():
        for kk, vv in v.items():
            for kkk, vvv in vv.items():
                if vvv is None:
                    for bi in range(batch):
                        new_As[bi][k] = new_As[bi].get(k, {})
                        new_As[bi][k][kk] = new_As[bi][k].get(kk, {})
                        new_As[bi][k][kk][kkk] = None 
                
                elif vvv.ndim == 0:
                    for bi in range(batch):
                        new_As[bi][k] = new_As[bi].get(k, {})
                        new_As[bi][k][kk] = new_As[bi][k].get(kk, {})
                        new_As[bi][k][kk][kkk] = vvv.cpu().clone()
                
                elif vvv.ndim > 0:
                    tmp = vvv.view(num_images, -1, *vvv.shape[1:])
                    for bi, a in enumerate(torch.chunk(tmp, tmp.shape[1], dim = 1)):
                        new_As[bi][k] = new_As[bi].get(k, {})
                        new_As[bi][k][kk] = new_As[bi][k].get(kk, {})
                        new_As[bi][k][kk][kkk] = a.squeeze(1).cpu() 
                
                else:
                    import pdb;pdb.set_trace()
    return new_As


def seperate_per_img(num_images, splits_per_example, dom_ub, dom_lb, dom_ub_point, dom_lb_all, dom_ub_all, lAs, slopes, split_history, betas, intermediate_betas, primals, dom_cs, A):

    dom_ub = dom_ub.view(num_images, -1, *dom_ub.shape[1:])
    dom_lb = dom_lb.view(num_images, -1, *dom_lb.shape[1:])
    
    batch = dom_lb.shape[1]
    
    dom_lb_all_new = [[] for _ in range(num_images)]
    dom_ub_all_new = [[] for _ in range(num_images)]
    
    for lb, ub in zip(dom_lb_all, dom_ub_all):
        for im, clb in enumerate(torch.chunk(lb, num_images, dim=0)):
            dom_lb_all_new[im].append(clb)
        
        for im, cub in enumerate(torch.chunk(ub, num_images, dim=0)):
            dom_ub_all_new[im].append(cub)
    
    lAs_new = [[] for _ in range(num_images)]
    for la in lAs:
        for im, l in enumerate(torch.chunk(la, num_images, dim=0)):
            lAs_new[im].append(l)
    
    slopes_new =  [{} for _ in range(num_images)]
    for k, v in slopes.items():
        for kk, vv in v.items():
            for im, sl in enumerate(torch.chunk(vv, num_images, dim=2)):
                    if k not in slopes_new[im]:
                        slopes_new[im][k] = {}
                    slopes_new[im][k][kk] = sl

    
    betas_new = [[[] for _ in range(batch) ] for j in range(num_images)]
    unexpanded_batch = (batch // 2)
    for i in range(len(betas)):
        for j, tens in enumerate(torch.chunk(betas[i], num_images)):
                for k in range(len(tens)):
                    betas_new[j][k].append(tens[k, :splits_per_example[k % unexpanded_batch, i]])

    dom_cs = dom_cs.view(num_images, -1, *dom_cs.shape[1:])
    
    new_As = seperate_As_per_batch(A, num_images, batch)

    return dom_ub, dom_lb, dom_ub_point, dom_lb_all_new, dom_ub_all_new, lAs_new, slopes_new, split_history, betas_new, intermediate_betas, primals, dom_cs, new_As
    

    

def expand_intermediate_layer_bounds(self, decision, batch, all_img_pre_lb, all_img_pre_ub):
    # pre_ub_all[:-1] means pre-set bounds for all intermediate layers
    #create a list of (2 * batch) intermediate layer bounds 
    with torch.no_grad():
            # Setting the neuron upper/lower bounds with a split to 0.
            intermediate_layer_bounds_lst = []
            for pre_lb_all, pre_ub_all in zip(all_img_pre_lb, all_img_pre_ub):
                zero_indices_batch = [[] for _ in range(len(pre_lb_all) - 1)]
                zero_indices_neuron = [[] for _ in range(len(pre_lb_all) - 1)]
                for i in range(batch):
                    d, idx = decision[i][0], decision[i][1]
                    # We save the batch, and neuron number for each split, and will set all corresponding elements in batch.
                    zero_indices_batch[d].append(i)
                    zero_indices_neuron[d].append(idx)
                zero_indices_batch = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_batch]
                zero_indices_neuron = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_neuron]

                # 2 * batch + diving_batch
                upper_bounds = [torch.cat([i[:batch], i[:batch], i[batch:]], dim=0) for i in pre_ub_all[:-1]]
                lower_bounds = [torch.cat([i[:batch], i[:batch], i[batch:]], dim=0) for i in pre_lb_all[:-1]]

                tmp_intermediate_layer_bounds = {}
                for d in range(len(lower_bounds)):
                    # for each layer except the last output layer
                    if len(zero_indices_batch[d]):
                        # we set lower = 0 in first half batch, and upper = 0 in second half batch
                        lower_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d], zero_indices_neuron[d]] = 0.0
                        upper_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d] + batch, zero_indices_neuron[d]] = 0.0
                    tmp_intermediate_layer_bounds[self.name_dict[d]] = [lower_bounds[d], upper_bounds[d]]
                intermediate_layer_bounds_lst.append(tmp_intermediate_layer_bounds)
            
            new_intermediate_layer_bounds = {}
            for key in intermediate_layer_bounds_lst[0]:
                new_intermediate_layer_bounds[key] = [[], []]
                for tmp_intermediate_layer_bounds in intermediate_layer_bounds_lst:
                    new_intermediate_layer_bounds[key][0].append(tmp_intermediate_layer_bounds[key][0])
                    new_intermediate_layer_bounds[key][1].append(tmp_intermediate_layer_bounds[key][1])
                
                new_intermediate_layer_bounds[key][0] = torch.cat(new_intermediate_layer_bounds[key][0])
                new_intermediate_layer_bounds[key][1] = torch.cat(new_intermediate_layer_bounds[key][1])
            
            return new_intermediate_layer_bounds
            
                
                
                