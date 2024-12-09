import multiprocessing.process
import torch
import numpy as np
import os
import itertools
from src.adaptiveRavenResult import Result, AdaptiveRavenResult
from raven.src.network_conversion_helper import get_pytorch_net
import raven.src.config as config
from auto_LiRPA.operators import BoundLinear, BoundConv
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from src.common import RavenMode
from src.gurobi_certifier import RavenLPtransformer, TopkLPtransformer
import src.util as util
import time
from src.metaResult import PropertyResult
import concurrent.futures 
import multiprocessing
import threading
from tqdm import tqdm
from copy import deepcopy
import signal
import sys
sys.path.append('src/abc/complete_verifier')
from beta_CROWN_solver import LiRPAConvNet
from batch_branch_and_bound import relu_bab_parallel
from batch_branch_and_bound_multiple import relu_bab_parallel_multiple
from bab_t2 import relu_bab_parallel_targetted
import arguments
from concurrent.futures import ThreadPoolExecutor
import random
from src.adaptiveRavenBackend import AdaptiveRavenBackend
from uapAttack import run_uap_attack
from collections import defaultdict
import math

class MetaScheduler:
    def __init__(self, prop, nets, args):
        self.args = args
        self.prop = prop 
        self.nets = nets
        self.device = args.device if torch.cuda.is_available else 'cpu'
        self.lock = threading.Lock()
        self.layer_names = []
        self.input_names = []
        self.final_names = []
        self.refined_bounds = {}
        self.refined_bounds_multi_ex = {}
        self.torch_models = []
        self.bounded_models = []
        self.final_layer_weights = []
        self.final_layer_biases = []
        self.property_result = PropertyResult()
        self.number_of_class = 10
        self.base_method = 'CROWN'
        self.optimized_method = 'CROWN-Optimized'
        self.lower_bnds_dict = {}
        self.lb_coef_dict, self.lb_bias_dict = {}, {}
        self.tuple_of_indices_cross_ex = {}
        self.tuple_of_refined_indices_cross_ex = {}
        self.cross_ex_loss = {}
        self.cross_ex_loss_dict = {}
        self.unverified_indices = torch.arange(len(self.prop.inputs))
        self.individual_verified = 0.0
        self.tuple_of_indices_cross_ex_branching = {}
        self.cross_ex_loss_branching = {}
        self.alpha_crown_bounds = {}

    def initialize_models(self):
        bound_opts = {'use_full_conv_alpha' : False}
        for net in self.nets:
            if not self.args.skip_torch_init:
                self.torch_models.append(get_pytorch_net(model=net, remove_last_layer=False, all_linear=False))
                self.torch_models[-1] = self.torch_models[-1].to(self.device)
                self.final_layer_weights.append(net[-1].weight.to(self.device))
                self.final_layer_biases.append(net[-1].bias.to(self.device))
            else:
                self.torch_models.append(net)
                self.torch_models[-1] = self.torch_models[-1].to(self.device)
                self.final_layer_weights.append(net.linear2.weight.to(self.device))
                self.final_layer_biases.append(net.linear2.bias.to(self.device))
            self.bounded_models.append(BoundedModule(self.torch_models[-1], (self.prop.inputs), bound_opts=bound_opts))
        
        self.populate_names()
        if self.args.refine_intermediate_bounds:
            assert self.args.optimize_layers_count is not None
            assert self.layer_names is not None
            length = min(self.args.optimize_layers_count, len(self.layer_names) - 1)
            self.optimize_layer_names = self.layer_names[-(length+1):-1]
            for model in self.bounded_models:
                model.set_optimize_layers_bounds(self.optimize_layer_names)
    
    def populate_names(self):
        for model in self.bounded_models:
            i = 0
            last_name = None
            for node_name, node in model._modules.items():
                if i == 0:
                    self.input_names.append(node_name)
                i += 1
                if type(node) in [BoundLinear, BoundConv]:
                    self.layer_names.append(node_name)
                    last_name = node_name
            assert last_name is not None
            self.final_names.append(node_name)
    
    def shift_to_device(self, device, models, prop, indices=None, refined_bounds=None):
        with self.lock:
            self.shift_props_to_device(prop=prop, device=device)
            if indices is not None:
                indices = indices.to(device)
            if refined_bounds is None:
                refined_bounds = self.refined_bounds
            for i, model in enumerate(models):
                models[i] = model.to(device) 
                # self.final_layer_weights[i] = self.final_layer_weights[i].to(device)
                # self.final_layer_biases[i].to(device)
            for _, element in self.refined_bounds.items():
                for x in element:
                    x = x.to(device)
    
    def shift_props_to_device(self, prop, device):
        prop.inputs = prop.inputs.to(device)
        prop.labels = prop.labels.to(device)
        prop.constraint_matrices = prop.constraint_matrices.to(device)
        prop.lbs = prop.lbs.to(device)
        prop.ubs = prop.ubs.to(device)
    
    
    @torch.no_grad()
    def run_crown_opt(self):
        indices = torch.arange(0, self.args.count_per_prop)
        if self.args.count_per_prop > self.args.overall_batch_size:
            indices_lst = list(torch.chunk(indices, math.ceil(len(indices)/self.args.refinement_batch_size)))
        else:
            indices_lst = [indices]
            
        
        overall_lA, overall_lbias, overall_lb = [], [], []
        for indices in indices_lst:
            
            self.ptb = PerturbationLpNorm(norm = np.inf, x_L=self.prop.lbs[indices], x_U=self.prop.ubs[indices])
            bounded_images = BoundedTensor(self.prop.inputs[indices], self.ptb)
            coef_dict = { self.final_names[0]: [self.input_names[0]]}
            for model in self.bounded_models:
                result = model.compute_bounds(x=(bounded_images,), method=self.base_method, C=self.prop.constraint_matrices[indices],
                                            bound_upper=False, return_A=True, needed_A_dict=coef_dict, 
                                            multiple_execution=False, execution_count=None, ptb=self.ptb, 
                                            unperturbed_images = self.prop.inputs[indices])
                lower_bnd, _, A_dict = result
                lA = A_dict[self.final_names[0]][self.input_names[0]]['lA']
                lbias = A_dict[self.final_names[0]][self.input_names[0]]['lbias']
                lA = torch.reshape(lA,(len(indices), self.number_of_class-1,-1))
                
                overall_lA.append(lA)
                overall_lbias.append(lbias)
                overall_lb.append(lower_bnd)
        return torch.cat(overall_lA), torch.cat(overall_lbias), torch.cat(overall_lb)
    

    
    def get_milp(self, lb_coef_dict, lb_bias_dict, non_verified_indices = None, topk_idx = None, ginf = 300):
        if topk_idx is not None:
             return TopkLPtransformer(eps=self.prop.eps, inputs=self.prop.inputs[topk_idx:topk_idx + 1], batch_size= 1,
                                         roll_indices=None, lb_bias= None, lb_coef= None,
                                         lb_coef_dict= lb_coef_dict, lb_bias_dict = lb_bias_dict, non_verified_indices= non_verified_indices,
                                         lb_penultimate_coef=None, lb_penultimate_bias=None, ub_penultimate_coef=None,
                                         ub_penultimate_bias=None, lb_penult=None, ub_penult=None,
                                         constraint_matrices=self.prop.constraint_matrices[topk_idx:topk_idx + 1],
                                         input_lbs=self.prop.lbs[topk_idx:topk_idx + 1], input_ubs=self.prop.ubs[topk_idx:topk_idx + 1], disable_unrolling=True).formulate_constriants_from_dict(final_weight= self.final_layer_weights[0],
                                                        final_bias= self.final_layer_biases[0])
            
        return RavenLPtransformer(eps=self.prop.eps, inputs=self.prop.inputs, batch_size=self.args.count_per_prop,
                                         roll_indices=None, lb_bias= None, lb_coef= None,
                                         lb_coef_dict= lb_coef_dict, lb_bias_dict = lb_bias_dict, non_verified_indices= non_verified_indices,
                                         lb_penultimate_coef=None, lb_penultimate_bias=None, ub_penultimate_coef=None,
                                         ub_penultimate_bias=None, lb_penult=None, ub_penult=None,
                                         constraint_matrices=self.prop.constraint_matrices, ginf= ginf,
                                         input_lbs=self.prop.lbs, input_ubs=self.prop.ubs, disable_unrolling=True).formulate_constriants_from_dict(final_weight= self.final_layer_weights[0],
                                                        final_bias= self.final_layer_biases[0])
    
    def initialize_for_parallel(self):
        bound_opts = {'use_full_conv_alpha' : self.args.full_alpha}
        
        num_for_paralel = self.args.maximum_cross_execution_count + 1
        
        self.devices = {k: self.device for k in range(2, num_for_paralel)}
        self.refined_bounds_multi_ex = {k: {} for k in range(2, num_for_paralel)}
        self.torch_models_for_parallel = {k: [] for k in range(2, num_for_paralel)}
        self.bounded_models_for_parallel = {k: [] for k in range(2, num_for_paralel)}

        
        for torch_model, bounded_model in zip(self.torch_models, self.bounded_models):
            for k in range(2, num_for_paralel):
                if k == 2:
                    self.torch_models_for_parallel[k].append(torch_model)
                    self.bounded_models_for_parallel[k].append(bounded_model)
                else:
                    self.torch_models_for_parallel[k].append(deepcopy(torch_model).to(self.devices[k]))
                    self.bounded_models_for_parallel[k].append(BoundedModule(self.torch_models_for_parallel[k][-1], (deepcopy(self.prop.inputs.to(self.devices[k]))), bound_opts= bound_opts))

    
    def store_refined_bounds(self):
        for model in self.bounded_models:
            for node_name, node in model._modules.items():
                if node_name in self.optimize_layer_names:
                    self.refined_bounds[node_name] = [node.lower.detach().clone(), node.upper.detach().clone()]
                    if self.args.parallelize_executions:
                        self.refined_bounds_multi_ex[2][node_name] = [node.lower.detach().clone().to(self.devices[2]), node.upper.detach().clone().to(self.devices[2])]
                        self.refined_bounds_multi_ex[3][node_name] = [node.lower.detach().clone().to(self.devices[3]), node.upper.detach().clone().to(self.devices[3])]
                        self.refined_bounds_multi_ex[4][node_name] = [node.lower.detach().clone().to(self.devices[4]), node.upper.detach().clone().to(self.devices[4])]
    
    def store_refined_batched(self, new_refined_bounds = None):
        for node_name, node in self.bounded_models[0]._modules.items():
            if node_name in self.optimize_layer_names:
                if node_name in self.refined_bounds:
                    if new_refined_bounds is not None:
                    
                        self.refined_bounds[node_name] = [torch.cat([self.refined_bounds[node_name][0], new_refined_bounds[node_name][0]]), torch.cat([self.refined_bounds[node_name][1], new_refined_bounds[node_name][1]])]
                        
                        if self.args.parallelize_executions:
                            for k in self.refined_bounds_multi_ex:
                                self.refined_bounds_multi_ex[k][node_name] = [torch.cat([self.refined_bounds_multi_ex[k][node_name][0], new_refined_bounds[node_name][0].to(self.devices[k])]), torch.cat([self.refined_bounds_multi_ex[k][node_name][1], new_refined_bounds[node_name][1].to(self.devices[k])])]
                    else:
                        self.refined_bounds[node_name] = [torch.cat([self.refined_bounds[node_name][0], node.lower.detach().clone()]), torch.cat([self.refined_bounds[node_name][1], node.upper.detach().clone()])]
                        
                        if self.args.parallelize_executions:
                            for k in self.refined_bounds_multi_ex:
                                self.refined_bounds_multi_ex[k][node_name] = [torch.cat([self.refined_bounds_multi_ex[k][node_name][0], node.lower.detach().clone().to(self.devices[k])]), torch.cat([self.refined_bounds_multi_ex[k][node_name][1], node.upper.detach().clone().to(self.devices[k])])]
                
                else:  
                    if new_refined_bounds is not None:
                        self.refined_bounds[node_name] = new_refined_bounds[node_name]

                        if self.args.parallelize_executions:
                            for k in self.refined_bounds_multi_ex:
                                self.refined_bounds_multi_ex[k][node_name] = [new_refined_bounds[node_name][0].to(self.devices[k]), new_refined_bounds[node_name][1].to(self.devices[k])]
                    else:
                        self.refined_bounds[node_name] = [node.lower.detach().clone(), node.upper.detach().clone()]

                        if self.args.parallelize_executions:
                            for k in self.refined_bounds_multi_ex:
                                self.refined_bounds_multi_ex[k][node_name] = [node.lower.detach().clone().to(self.devices[k]), node.upper.detach().clone().to(self.devices[k])]
    
    def get_verified_count(self, lower_bnd):
        return torch.sum(lower_bnd.detach().cpu().min(axis=1)[0] > 0).numpy() if isinstance(lower_bnd, torch.Tensor) else sum([(lb.detach().cpu().min() > 0).item() for lb in lower_bnd])
    
    def reset_refined_bounds(self):
        for model in self.bounded_models:
            for node_name, node in model._modules.items():
                if node_name in self.optimize_layer_names:
                    del self.refined_bounds[node_name]
                
                    for k in self.refined_bounds_multi_ex:
                        del self.refined_bounds_multi_ex[k][node_name]
    
    
    def populate_cross_indices(self, cross_executional_indices, count, populate_tuples=False):
        indices = cross_executional_indices
        final_indices, tuple_indices = util.generate_indices(indices=indices, threshold= float('inf'), count=count, use_entries= False)
        # print(f'ex count {count} tuple list {tuple_indices} cross ex indices {final_indices}')
        if populate_tuples:
            if count in self.tuple_of_indices_cross_ex:
                self.tuple_of_indices_cross_ex[count].extend(tuple_indices)
            else: 
                self.tuple_of_indices_cross_ex[count] = tuple_indices
        else:
            for i in range(len(tuple_indices)):
                tup = []
                for j in range(len(tuple_indices[i])):
                    tup.append(tuple_indices[i][j].item())
                tuple_indices[i] = tuple(tup)
            
            if count in self.tuple_of_refined_indices_cross_ex:
                self.tuple_of_refined_indices_cross_ex[count].extend(tuple_indices)
            else: 
                self.tuple_of_refined_indices_cross_ex[count] = tuple_indices
        return final_indices
    
    def select_indices(self, lower_bound, threshold=None, lb_threshold=None):
        if type(lower_bound) is list:
            indices = []
            for i in range(len(lower_bound)):
                if self.select_indices(lower_bound[i], threshold, lb_threshold).tolist() != []:
                    indices.append(i)  
            return torch.as_tensor(indices)
        min_logit_diff = lower_bound.detach().cpu().min(axis=1)[0]
        min_logit_diff_sorted = min_logit_diff.sort(descending=True)
        # print(f'sorted logit diff {min_logit_diff_sorted[0]}')
        if lb_threshold is None:
            indices = min_logit_diff_sorted[1][(min_logit_diff_sorted[0] < 0.0)]
        else:
            indices = min_logit_diff_sorted[1][torch.logical_and((min_logit_diff_sorted[0] < 0.0), (min_logit_diff_sorted[0] >= lb_threshold))]
        length = indices.shape[0]
        if threshold is not None:
            indices = indices[:min(length, threshold)]
        # print(f'filtered min_indices {min_logit_diff[indices]}')
        return indices
    
    def store_alpha_bounds(self, indices, lower_bnds):
        for i, idx in enumerate(indices.tolist()):
            self.alpha_crown_bounds[idx] = lower_bnds[i]
    

    def run_refinement_incomplete(self, indices, device, multiple_execution=False, execution_count=None, iteration=None, 
                       indices_for_refined_bounds=None, refine_intermediate_bounds=False, populate_results=True, 
                       models=None, prop=None, refined_bounds=None, topk = None):
        if models is None:
            models = self.bounded_models
        if prop is None:
            prop = self.prop
        if refined_bounds is None:
            refined_bounds = self.refined_bounds
        if self.args.skip_torch_init:
            indices_for_refined_bounds = None
        indices_lst, indices_for_refined_bounds_lst = [], []
        if len(indices) > self.args.refinement_batch_size:
            execution_count = execution_count if execution_count is not None else 1 
            
            indices = indices.view(execution_count, -1).T
            if indices_for_refined_bounds is not None:
                indices_for_refined_bounds = indices_for_refined_bounds.view(execution_count, -1).T 
            
            max_rows = self.args.refinement_batch_size//execution_count
            for j in range(0, indices.shape[0], max_rows):
                indices_lst.append(indices[j:(j + max_rows)].T.flatten())
                if indices_for_refined_bounds is not None:
                    indices_for_refined_bounds_lst.append(indices_for_refined_bounds[j:(j + max_rows)].T.flatten())
                else:
                    indices_for_refined_bounds_lst.append(None)

        else:
            indices_lst = [indices]
            indices_for_refined_bounds_lst = [indices_for_refined_bounds]
        
        overall_lA, overall_lbias, overall_lb = [], [], []
        
        for indices, indices_for_refined_bounds in zip(indices_lst, indices_for_refined_bounds_lst):
            filtered_inputs = prop.inputs[indices]
            filtered_lbs, filtered_ubs = prop.lbs[indices], prop.ubs[indices]
            filtered_ptb = PerturbationLpNorm(norm = np.inf, x_L=filtered_lbs, x_U=filtered_ubs)
            filtered_dict = {}
            for key, element in refined_bounds.items():
                if indices_for_refined_bounds is None:
                    continue
                if key not in filtered_dict.keys():
                    filtered_dict[key] = []
                for x in element:
                    t = x[indices_for_refined_bounds]
                    filtered_dict[key].append(t)
                    
            bounded_images = BoundedTensor(filtered_inputs, filtered_ptb)
            filtered_constraint_matrices = prop.constraint_matrices[indices]
            coef_dict = {self.final_names[0]: [self.input_names[0]]}
            cross_ex_result = {}
            for model in models:
                result = model.compute_bounds(x=(bounded_images,), method=self.optimized_method, C=filtered_constraint_matrices,
                                            bound_upper=False, return_A=True, needed_A_dict=coef_dict,
                                            multiple_execution=multiple_execution, execution_count=execution_count, ptb=filtered_ptb, 
                                            unperturbed_images = filtered_inputs, iteration=iteration, 
                                            baseline_refined_bound=filtered_dict, 
                                            intermediate_bound_refinement= self.args.use_ib_refinement,
                                            always_correct_cross_execution=self.args.always_correct_cross_execution,
                                            cross_refinement_results=cross_ex_result,
                                            populate_trace=self.args.populate_trace, topk = topk)
                lower_bnd, _, A_dict = result
                lower_bnd = lower_bnd.cpu()
                lA = A_dict[self.final_names[0]][self.input_names[0]]['lA']
                lbias = A_dict[self.final_names[0]][self.input_names[0]]['lbias'].cpu()
                lA = torch.reshape(lA,(filtered_inputs.shape[0], self.number_of_class-1,-1)).cpu()
                if (multiple_execution == False) and (not self.args.skip_torch_init):
                    self.store_refined_batched()

                overall_lA.append(lA)
                overall_lbias.append(lbias)
                overall_lb.append(lower_bnd)
                
            
            if multiple_execution:
                if execution_count in self.cross_ex_loss:
                    self.cross_ex_loss[execution_count] = torch.cat([self.cross_ex_loss[execution_count], cross_ex_result['final_loss']])
                else:
                    self.cross_ex_loss[execution_count] = cross_ex_result['final_loss']
                cross_ex_loss = cross_ex_result['final_loss'].repeat(execution_count).cpu()
                
            else:
                cross_ex_loss = None
            
            if populate_results:
                self.store_alpha_bounds(indices, lower_bnd)
                self.populate_coef_and_bias(indices=indices, lb_coef=lA, lb_bias=lbias, lower_bnd=lower_bnd.min(axis=1)[0], cross_ex_loss= cross_ex_loss)

        return torch.cat(overall_lA), torch.cat(overall_lbias), torch.cat(overall_lb)  
    
    
    def run_cross_executional_refinement(self, count, indices_for, indices_for_refined_bounds, complete_verification = False, num_combs = None, meta_timeout = None):

        if self.args.parallelize_executions:
            models = self.bounded_models_for_parallel[count]
            device = self.devices[count]
            refined_bound = self.refined_bounds_multi_ex[count]
            torch_models = self.torch_models_for_parallel[count]
        else:
            models = self.bounded_models
            device = self.device
            refined_bound = self.refined_bounds
            torch_models = self.torch_models
        
        if complete_verification:
            return self.run_refinement_bab(indices = indices_for, device=device,
                                    multiple_execution=True, execution_count=count, iteration=self.args.refinement_iterations,
                                    indices_for_refined_bounds=indices_for_refined_bounds,
                                    refine_intermediate_bounds=self.args.refine_intermediate_bounds, models=models, refined_bounds=refined_bound, torch_models= torch_models, num_combs = num_combs, meta_timeout = meta_timeout)

        self.run_refinement_incomplete(indices= indices_for, device=device,
                                    multiple_execution=True, execution_count=count, iteration=self.args.refinement_iterations,
                                    indices_for_refined_bounds=indices_for_refined_bounds,
                                    refine_intermediate_bounds=self.args.refine_intermediate_bounds, models=models, refined_bounds=refined_bound)
            
    
    
    
    
    def get_post_refinement_unverified_indices(self):
        final_indices = []
        for i in self.unverified_indices:
            if i not in self.cross_executional_indices_from_refinement:
                # print(f'baseline lower {self.baseline_lowerbound[i].min()}')
                if self.args.lp_threshold is not None and self.unverified_lb[i].min() <= self.args.lp_threshold:
                    # print(f'filtered index with lb {self.baseline_lowerbound[i]}')
                    continue
                final_indices.append(i)
            else:
                final_indices.append(i)
        return torch.tensor(final_indices, device='cpu')
    
    
    def populate_coef_and_bias(self, indices, lb_coef, lb_bias, lower_bnd, cross_ex_loss = None):
        assert len(indices) == lb_coef.shape[0]
        assert len(indices) == lb_bias.shape[0]
        assert len(indices) == lower_bnd.shape[0]
        if cross_ex_loss is not None:
            assert len(indices) == cross_ex_loss.shape[0]
        else:
            cross_ex_loss = torch.zeros_like(lower_bnd) - 1e9
        lb_coef = lb_coef.detach()
        lb_bias = lb_bias.detach()
        for i, ind in enumerate(indices):
            if type(ind) is torch.Tensor:
                index = ind.item()
            else:
                index = ind
            if index not in self.lb_bias_dict.keys():
                self.lb_coef_dict[index] = []
                self.lb_bias_dict[index] = []
                self.lower_bnds_dict[index] = []
                self.cross_ex_loss_dict[index] = []
            self.lb_bias_dict[index].append(lb_bias[i])
            self.lb_coef_dict[index].append(lb_coef[i])
            self.lower_bnds_dict[index].append(lower_bnd[i])
            self.cross_ex_loss_dict[index].append(cross_ex_loss[i])

    def select_apprx(self, method, num_approx):
        assert method in ['bounds', 'loss']
        if method == 'bounds':
            store = self.lower_bnds_dict
        else:
            store = self.cross_ex_loss_dict
        
        selected_lb_coef_dict = {}
        selected_lb_bias_dict = {}
        
        for index in range(self.args.count_per_prop):
            if num_approx is not None and len(self.lb_bias_dict[index]) > num_approx: 
                selected_approx_indices = torch.stack(store[index]).topk(num_approx).indices
                selected_lb_coef_dict[index] = []
                selected_lb_bias_dict[index] = []

                for i in selected_approx_indices:
                    selected_lb_coef_dict[index].append(self.lb_coef_dict[index][i])
                    selected_lb_bias_dict[index].append(self.lb_bias_dict[index][i].clone())
            
            else:
                selected_lb_coef_dict[index] = self.lb_coef_dict[index]
                selected_lb_bias_dict[index] = [tens.clone() for tens in self.lb_bias_dict[index]]
        
        return selected_lb_coef_dict, selected_lb_bias_dict       
                    

    
    def prune_linear_apprx(self, ind):
        new_coef_list = []
        new_bias_list = []
        new_lb_list = []
        min_lb = min(self.lower_bnds_dict[ind])
        for i in range(len(self.lower_bnds_dict[ind])):
            if self.lower_bnds_dict[ind][i] > min_lb:
                new_bias_list.append(self.lb_bias_dict[ind][i])
                new_coef_list.append(self.lb_coef_dict[ind][i])
                new_lb_list.append(self.lower_bnds_dict[ind][i])
        if len(new_lb_list) > 0:
            self.lb_coef_dict[ind] = new_coef_list
            self.lb_bias_dict[ind] = new_bias_list
            self.lower_bnds_dict[ind] = new_lb_list
    
    def get_verified_tuples(self, max_execution_count, cross_ex_loss, tuple_of_indices_cross_ex):
        cross_verified = []
        for i in range(2, max_execution_count+1):
            if i not in cross_ex_loss.keys() or i not in tuple_of_indices_cross_ex.keys():
                continue
                
            if len(cross_ex_loss[i]) > len(tuple_of_indices_cross_ex[i]):
                raise RuntimeError
            
            for j in range(len(cross_ex_loss[i])):
                if cross_ex_loss[i][j] >= 0:
                    cross_verified.append(tuple_of_indices_cross_ex[i][j])
        
        return cross_verified

    def select_indices_branching2(self):
        all_tuples = []
        all_refined_tuples = []
        all_scores = []
        unverified_indices = set(self.unverified_indices.tolist())
        pos_indices = set()
        

        for execution_count, tup_lst in self.tuple_of_indices_cross_ex.items():
            self.tuple_of_indices_cross_ex_branching[execution_count] = []
            self.cross_ex_loss_branching[execution_count] = []

            for i, tup in enumerate(tup_lst):
                if all((elem in unverified_indices)  for elem in tup) and self.cross_ex_loss[execution_count][i] >= 0:
                    #if score > 0 add them for final cross ex loss computation but not for branching
                    self.tuple_of_indices_cross_ex_branching[execution_count].append(tup)
                    self.cross_ex_loss_branching[execution_count].append(self.cross_ex_loss[execution_count][i].item())
                    pos_indices.update(tup)
        iter = 0
        for execution_count, tup_lst in self.tuple_of_indices_cross_ex.items():
            for i, tup in enumerate(tup_lst):
                #consider only the tuples with unverified indices
                if all((elem in unverified_indices) for elem in tup) and (self.cross_ex_loss[execution_count][i] < 0) and (execution_count in self.args.branching_execution_counts):
                    all_tuples.append(tup)
                    all_refined_tuples.append(self.tuple_of_refined_indices_cross_ex[execution_count][i])
                    all_scores.append(self.cross_ex_loss[execution_count][i].item())
                    iter += 1
        if len(all_tuples) == 0:
            return [], [], [], [], []      
        
        shift = min(all_scores) - self.args.bias
        orig_scores = deepcopy(all_scores)   
        for i, _ in enumerate(all_tuples):
            all_scores[i] = (all_scores[i] - shift)
        
        signal.signal(signal.SIGALRM, util.timeout_handler)
        if self.args.greedy_tuple:
            selected_tuples = util.top_tuple(all_tuples, all_scores)
        else:
            try:
                signal.alarm(self.args.timeout_seconds)
                selected_tuples = util.max_disjoint_tuples(all_tuples, all_scores, 10) 
                signal.alarm(0)
            except:
                selected_tuples = util.greedy_disjoint_tuples(all_tuples, all_scores, 10) 
        assert len(selected_tuples) == len(set(selected_tuples))
        selected_refined_tuples = [all_refined_tuples[all_tuples.index(tup)] for tup in selected_tuples]
        
        return all_tuples, all_refined_tuples, orig_scores, *self.process_selected_tuples(selected_tuples, selected_refined_tuples)
    
    
    
    def update_selection_branching(self, all_tuples, all_refined_tuples, all_scores, verified_tuples, unverified_tuples):
        for unverified_tuple in unverified_tuples:
            try:
                idx = all_tuples.index(unverified_tuple)
            except:
                import pdb;pdb.set_trace()
            all_tuples.pop(idx)
            all_refined_tuples.pop(idx)
            all_scores.pop(idx)
        
        if len(all_tuples) == 0:
            return [], [], [], [], []
        
        for verified_tuple in verified_tuples:
            try:
                idx = all_tuples.index(verified_tuple)
            except:
                import pdb;pdb.set_trace()
            all_tuples.pop(idx)
            all_refined_tuples.pop(idx)
            all_scores.pop(idx)
        
        if len(all_tuples) == 0:
            return [], [], [], [], []

        shift = min(all_scores) - self.args.bias
        orig_scores = deepcopy(all_scores)
        
        for i in range(len(all_scores)):
            all_scores[i] = (all_scores[i] - shift)
        
        signal.signal(signal.SIGALRM, util.timeout_handler)
        
        if self.args.greedy_tuple:
            selected_tuples = util.top_tuple(all_tuples, all_scores)
        else:
            try:
                signal.alarm(self.args.timeout_seconds)
                selected_tuples = util.max_disjoint_tuples(all_tuples, all_scores, 10) 
                signal.alarm(0)
            except:
                selected_tuples = util.greedy_disjoint_tuples(all_tuples, all_scores, 10)  

        assert len(selected_tuples) == len(set(selected_tuples))
        selected_refined_tuples = [all_refined_tuples[all_tuples.index(tup)] for tup in selected_tuples]
        
        return all_tuples, all_refined_tuples, orig_scores, *self.process_selected_tuples(selected_tuples, selected_refined_tuples)
        
                

    
    def process_selected_tuples(self, selected_tuples, selected_refined_tuples):
        tup_dct = defaultdict(list)
        refined_tup_dct = defaultdict(list)
        
        for tup, rtup in zip(selected_tuples, selected_refined_tuples):
            assert len(tup) == len(rtup)
            execution_count = len(tup)
            self.tuple_of_indices_cross_ex_branching[execution_count].append(tup)
            
            tup_dct[execution_count].append(tup)
            refined_tup_dct[execution_count].append(rtup)
        
        tup_dct, refined_tup_dct = dict(tup_dct), dict(refined_tup_dct)
        
        final_tuples = []
        final_refined_tuples = []
        
        for execution_count in tup_dct:
            result = []
            refined_result = []
            sublist = []
            refined_sublist = []
            
            for tup, rtup in zip(tup_dct[execution_count], refined_tup_dct[execution_count]):
                sublist.append(tup)
                refined_sublist.append(rtup)
                if len(sublist) == self.args.max_branching_execution_count_dct[execution_count]:
                    result.append(torch.tensor(sublist).view(-1, execution_count).T.flatten())
                    refined_result.append(torch.tensor(refined_sublist).view(-1, execution_count).T.flatten())
                    sublist = []
                    refined_sublist = []
            
            if sublist:
                result.append(torch.tensor(sublist).view(-1, execution_count).T.flatten())
                refined_result.append(torch.tensor(refined_sublist).view(-1, execution_count).T.flatten())
            
            for (tup, rtup) in zip(result, refined_result):
                final_tuples.append((execution_count, tup))
                final_refined_tuples.append((execution_count, rtup))
        
        return final_tuples, final_refined_tuples

    

    
    
    def cross_ex_refinement(self, refinement_indices = None, lower_bnd = None, complete_verification = False):
        assert complete_verification == False
        total_length = min(max(self.args.execution_count_dct.keys()), len(self.unverified_indices))
        if self.args.parallelize_executions:
            executions = []
            for i in range(2, total_length + 1):
                cross_executional_indices_from_refinement = self.select_indices(lower_bound= lower_bnd, threshold = self.args.execution_count_dct[i])
                
                cross_executional_indices = refinement_indices[cross_executional_indices_from_refinement].detach().cpu().numpy()
                
                tmp1 = self.populate_cross_indices(cross_executional_indices=cross_executional_indices, count=i, populate_tuples=True).to(self.devices[i]),
                tmp2 = self.populate_cross_indices(cross_executional_indices= cross_executional_indices_from_refinement,
                                                                            count=i).to(self.devices[i])
                executions.append(self.run_cross_executional_refinement, i, tmp1, tmp2, complete_verification)

            
            with ThreadPoolExecutor(max_workers= total_length) as executor:
                [executor.submit(execution) for execution in executions]
        else:
            for i in range(2, total_length + 1):
                cross_executional_indices_from_refinement = self.select_indices(lower_bound= lower_bnd, threshold =  self.args.execution_count_dct[i])
                cross_executional_indices = refinement_indices[cross_executional_indices_from_refinement].detach().cpu().numpy()
                tmp1 = self.populate_cross_indices(cross_executional_indices=cross_executional_indices, count=i, populate_tuples=True).to(self.device)
                tmp2 = self.populate_cross_indices(cross_executional_indices= cross_executional_indices_from_refinement,
                                                                            count=i).to(self.device)
                self.run_cross_executional_refinement(i, tmp1, tmp2, complete_verification)

        
    def get_unverified_indices(self, lower_bnd):
        idxs = (lower_bnd.detach().cpu().min(axis=1)[0] < 0).nonzero().squeeze()
        if idxs.ndim == 0 and idxs.numel() == 1:
            idxs = idxs.unsqueeze(0)
        return idxs
        

    def verify_crown(self):
        lA, lbias, lower_bnd = self.run_crown_opt()
        dt = time.time() - self.meta_start
        lA, lbias, lower_bnd = lA.detach().cpu(), lbias.detach().cpu(), lower_bnd.detach().cpu() 
        self.individual_verified = self.get_verified_count(lower_bnd=lower_bnd)
        self.unverified_indices = self.get_unverified_indices(lower_bnd)
        
        
        crown_ceritified_accuracy = self.individual_verified / self.args.count_per_prop * 100
        self.property_result.verified_dict[dt] = (crown_ceritified_accuracy, 'CROWN')
        self.property_result.alg_results['CROWN']['accuracy'] = crown_ceritified_accuracy
        self.property_result.alg_results['CROWN']['time'] = dt
        
        
        all_indices = [i for i in range(self.prop.inputs.shape[0])]
        self.populate_coef_and_bias(all_indices,  lA, lbias, lower_bnd.min(axis=1)[0])
        return lA, lbias, lower_bnd
    
    def verify_alpha_crown(self, topk = None):
        lA, lbias, lower_bnd = self.run_refinement_incomplete(self.unverified_indices, self.device, 
                                                              topk = topk)
        dt = time.time() - self.meta_start

            
        lA, lbias, lower_bnd = lA.detach(), lbias.detach(), lower_bnd.detach() 
        unverified_alpha_crown_indices = self.get_unverified_indices(lower_bnd)
        
        refinement_indices = self.unverified_indices
        alpha_crown_lb = lower_bnd
        
        self.unverified_indices = self.unverified_indices[unverified_alpha_crown_indices]
        
        alpha_crown_ver = self.get_verified_count(lower_bnd)
        max_uap = (self.individual_verified + alpha_crown_ver)/(self.args.count_per_prop) * 100.0
        
        self.individual_verified += self.get_verified_count(lower_bnd)
        self.property_result.verified_dict[dt] = (max_uap, 'alpha-CROWN')
        self.property_result.alg_results['alpha-CROWN']['accuracy'] = max_uap
        self.property_result.alg_results['alpha-CROWN']['time'] = dt
        return refinement_indices, alpha_crown_lb
    
    def solv_async_milp(self, milp_verifier, individual_verified, algorithm, cross_ex_verified_tuples = None):
        org_start = deepcopy(self.meta_start)
        milp_verr = milp_verifier.solv_MILP(cross_ex_verified_tuples= cross_ex_verified_tuples)
        print('MILP FROM ', algorithm)
        dt = time.time() - org_start
        if dt < self.args.total_time:
            uap_accuracy = (individual_verified + milp_verr)/(self.args.count_per_prop) * 100
            self.property_result.verified_dict[dt] = (uap_accuracy, algorithm)
            self.property_result.alg_results[algorithm]['accuracy'] = max(uap_accuracy, self.property_result.alg_results[algorithm].get('accuracy', 0))
            self.property_result.alg_results[algorithm]['time'] = max(dt, self.property_result.alg_results[algorithm].get('time', 0))

    
    def run_racoon(self, refinement_indices, alpha_crown_lb):
        self.cross_ex_refinement(refinement_indices, alpha_crown_lb)
        self.property_result.incomplete_losses = self.cross_ex_loss
        self.property_result.incomplete_tuples = self.tuple_of_indices_cross_ex
        
        selected_lAs, selected_lbiases = self.select_apprx('bounds', self.args.max_linear_apprx)    
        milp_verifier = self.get_milp(selected_lAs, selected_lbiases, self.unverified_indices.clone())
        return milp_verifier
    
    def generate_logit_combs(self, indices):
        lst = []
        for key in indices.tolist():
            unverified_logit_ids = torch.where(self.alpha_crown_bounds[key] < 0)[0]
            unverified_logits = self.alpha_crown_bounds[key][unverified_logit_ids]
            unverified_logits = sorted([(idx, logt) for idx, logt in zip(unverified_logit_ids.tolist(), 
                                        unverified_logits.tolist())], key = lambda x : x[1])
            lst.append([idx for idx, _ in unverified_logits])
        return list(itertools.product(*lst))
        

        
    def run_refinement_bab(self, indices, device, multiple_execution=False,
                    execution_count=None, iteration=None, 
                    indices_for_refined_bounds=None, refine_intermediate_bounds=False,
                    populate_results=True, models=None, prop=None, refined_bounds=None, torch_models = None, target = None, target_biases=None, selected_lbiases = None, num_combs = None, meta_timeout = None):
        
        if models is None:
            models = self.bounded_models
        if prop is None:
            prop = self.prop
        if refined_bounds is None:
            refined_bounds = self.refined_bounds
        if torch_models is None:
            torch_models = self.torch_models
        if self.args.skip_torch_init:
            indices_for_refined_bounds = None
        
        
        #filter based off of indices
        filtered_inputs = prop.inputs[indices]
        filtered_constraint_matrices = prop.constraint_matrices[indices]
        filtered_lbs, filtered_ubs = prop.lbs[indices], prop.ubs[indices]
        filtered_ptb = PerturbationLpNorm(norm = np.inf, x_L=filtered_lbs, x_U=filtered_ubs)
        
        domain = torch.stack([filtered_lbs.squeeze(0), filtered_ubs.squeeze(0)], dim=-1)
        bounded_images = BoundedTensor(filtered_inputs, filtered_ptb)
        max_initial_domains = arguments.Config['bab']['initial_max_domains']
        
        filtered_dict = {}
        for key, element in refined_bounds.items():
            if indices_for_refined_bounds is None:
                continue
            if key not in filtered_dict.keys():
                filtered_dict[key] = []
            for x in element:
                t = x[indices_for_refined_bounds]
                filtered_dict[key].append(t)
        coef_dict = {self.final_names[0]: [self.input_names[0]]}
        
        cross_ex_results = {}
        
        if target is not None:
            is_verified = True
            key = indices[0].item()
            lbiases = torch.zeros_like(target_biases, device = 'cpu')
            unverified_logits = torch.where(self.alpha_crown_bounds[key] < 0)[0].tolist()
            for j in range(0, len(unverified_logits), max_initial_domains):
                logts = unverified_logits[j : j + max_initial_domains]
                model_params = dict(model_ori = torch_models[0], in_size = filtered_inputs.shape, c = filtered_constraint_matrices[:, logts, :], device = models[0].device, input = filtered_inputs, full_alpha = self.args.full_alpha)
                refine_params = dict(needed_A_dict=coef_dict,
                                        multiple_execution=multiple_execution, execution_count=execution_count, ptb= filtered_ptb, 
                                        unperturbed_images = filtered_inputs, iteration=iteration, 
                                        baseline_refined_bound=filtered_dict, 
                                        intermediate_bound_refinement=  self.args.use_ib_refinement,
                                        always_correct_cross_execution=self.args.always_correct_cross_execution,
                                        cross_refinement_results= cross_ex_results,
                                        populate_trace=self.args.populate_trace, index = indices)
                
                ver, targetted_losses = relu_bab_parallel_targetted(model_params, refine_params, domain, bounded_images, 
                                                    target_lAs = target[:, logts, :], target_biases=target_biases[:, logts], 
                                                    t_scale = self.args.targetted_t_scale, filtered_lbs = filtered_lbs, filtered_ubs = filtered_ubs)
                lbiases[:, logts] = targetted_losses.cpu()
                is_verified = is_verified and ver
            
            if is_verified:
                self.unverified_indices = self.unverified_indices[self.unverified_indices != key]
                self.individual_verified += 1
            if self.args.max_targetted == 0:
                return
            selected_lbiases[key] = [selected_lbiases[key][i - 1] + lbiases[i] for i in range(1, len(lbiases))] + [selected_lbiases[key][i] for i in range(len(lbiases) - 1, len(selected_lbiases[key]))]
            return lbiases

        elif len(indices) == 1:
                key = indices[0].item()
                unverified_logit_ids = torch.where(self.alpha_crown_bounds[key] < 0)[0]
                unverified_logits = self.alpha_crown_bounds[key][unverified_logit_ids]
                unverified_logits = sorted([(idx, logt) for idx, logt in zip(unverified_logit_ids.tolist(), 
                                            unverified_logits.tolist())], key = lambda x : x[1])
                unverified_logits = [idx for idx, _ in unverified_logits]
                for j in range(0, len(unverified_logits), max_initial_domains):
                    logts = unverified_logits[j : j + max_initial_domains]
                    model = LiRPAConvNet(torch_models[0], filtered_inputs.shape, c = filtered_constraint_matrices[:, logts, :], 
                                         device = models[0].device, input = filtered_inputs, full_alpha = self.args.full_alpha)
                    _, _, _, _, is_verified, _ = relu_bab_parallel(model, domain, bounded_images, None)
                    if is_verified != 'safe':
                        return False
                return True

                      
        else:
            logit_combs = self.generate_logit_combs(indices)
            min_loss = torch.tensor([1e9 for _ in range(num_combs)])
            for logit_comb in logit_combs:
                comb_c = torch.cat([filtered_constraint_matrices[i : i + 1, logt : logt + 1, :] for i, logt in enumerate(logit_comb)])
                model = LiRPAConvNet(torch_models[0], filtered_inputs.shape, c = comb_c, device = models[0].device, input = filtered_inputs, full_alpha = self.args.full_alpha)
                
                model.init_refinement_params(needed_A_dict=coef_dict,
                                        multiple_execution=multiple_execution, execution_count=execution_count, ptb= filtered_ptb, 
                                        unperturbed_images = filtered_inputs, iteration=iteration, 
                                        baseline_refined_bound=filtered_dict, 
                                        intermediate_bound_refinement=  self.args.use_ib_refinement,
                                        always_correct_cross_execution=self.args.always_correct_cross_execution,
                                        cross_refinement_results= cross_ex_results,
                                        populate_trace=self.args.populate_trace, index = indices)
                
                cross_ex_losses = relu_bab_parallel_multiple(model, domain, bounded_images, indices = indices, num_combs = num_combs, meta_timeout = meta_timeout, t_scale = self.args.multiple_t_scale)
                if torch.any(cross_ex_losses < 0):
                    if multiple_execution:
                        self.cross_ex_loss_branching[execution_count] += cross_ex_losses.tolist()
                    return
                
                min_loss = torch.stack([min_loss, cross_ex_losses]).min(dim = 0).values
                
            if multiple_execution:
                self.cross_ex_loss_branching[execution_count] += min_loss.tolist()

    def run_abc(self, prune_before_refine = None):
        #self.refinement_indices = self.unverified_indices
        unverified_indices = self.unverified_indices.tolist()
        l_bnds = sorted([(i, self.lower_bnds_dict[i][-1]) for i in unverified_indices], key = lambda x : x[1], reverse= True)
        unverified_indices = [i[0] for i in l_bnds]
        
        if prune_before_refine is not None and prune_before_refine > 0:
            unverified_indices = unverified_indices[:prune_before_refine]
        
        unverified = set(unverified_indices)
        max_uap = 0.0
            
        for i in unverified_indices:
            if i not in unverified:
                continue     
            else:
                is_verified = self.run_refinement_bab(torch.tensor([i]),device = self.device)
                if is_verified: 
                    unverified.remove(i)
                    self.individual_verified = self.individual_verified + 1.0
                    dt = time.time() - self.meta_start 
                    self.property_result.verified_dict[dt] = ((self.individual_verified)/(self.args.count_per_prop) * 100.0, 'Individual Verified')
            max_uap = max(max_uap, ((self.individual_verified)/len(self.prop.inputs)) * 100 )      
            if len(unverified) == 0:
                return True, [], max_uap
            
        self.unverified_indices = torch.as_tensor(list(unverified), device = self.unverified_indices.device)
        self.property_result.alg_results['Individual Verified']['accuracy'] = (self.individual_verified)/(self.args.count_per_prop) * 100.0
        self.property_result.alg_results['Individual Verified']['time'] = time.time() - self.meta_start
        return max_uap

    def topk_strong_bounding(self, indices, topk):
        prop = self.prop
        filtered_inputs = prop.inputs[indices]
        filtered_constraint_matrices = prop.constraint_matrices[indices]
        filtered_lbs, filtered_ubs = prop.lbs[indices], prop.ubs[indices]
        filtered_ptb = PerturbationLpNorm(norm = np.inf, x_L=filtered_lbs, x_U=filtered_ubs)
        
        domain = torch.stack([filtered_lbs.squeeze(0), filtered_ubs.squeeze(0)], dim=-1)
        bounded_images = BoundedTensor(filtered_inputs, filtered_ptb)
        
        filtered_dict = {}
        coef_dict = {self.final_names[0]: [self.input_names[0]]}
        
        cross_ex_results = {}
        
        lirpa_models = [LiRPAConvNet(self.torch_models[i], filtered_inputs.shape, c = filtered_constraint_matrices, device = filtered_inputs.device, input = filtered_inputs, full_alpha = self.args.full_alpha) for i, model in enumerate(self.torch_models)]
        result = relu_bab_parallel(lirpa_models[0], domain, bounded_images, optimize_layer_names=None, 
                                    timeout=self.args.bab_timeout,
                                    ptb=filtered_ptb, topk=topk)
        _, _, lower_bnd, A_dict, spec_tree, new_refined_bounds = result
        topk_losses = spec_tree.get_topk_loss()
        return topk_losses
            
    def targetted_refinement(self, refinement_indices, alpha_crown_lb, milp_tasks, topk=2):
        cross_executional_indices_from_refinement = self.select_indices(lower_bound= alpha_crown_lb, threshold = self.args.num_targetted)
        cross_executional_indices = refinement_indices[cross_executional_indices_from_refinement].detach().cpu().numpy()
        if self.args.run_topk:
            selected_lAs, selected_lbiases = self.select_apprx('bounds', self.args.max_linear_apprx) 
        else:
            selected_lAs, selected_lbiases = self.select_apprx('loss', self.args.max_linear_apprx) 
        for i, j in zip(cross_executional_indices.tolist(), cross_executional_indices_from_refinement.tolist()):
            target_lAs = torch.stack([torch.zeros_like(selected_lAs[i][0])] + selected_lAs[i][:self.args.max_targetted]).to(self.device)
            target_lbiases = torch.stack([torch.zeros_like(selected_lbiases[i][0])] + selected_lbiases[i][:self.args.max_targetted]).to(self.device)  
            
            if self.args.run_topk:
                ewf = self.run_refinement_bab(torch.tensor([i] * len(target_lAs)),device = self.device, target = target_lAs, target_biases= target_lbiases, selected_lbiases = selected_lbiases)
                # import pdb; pdb.set_trace()
                milp_verifier = self.get_milp({0: selected_lAs[i][0].unsqueeze(0).clone()}, {0: selected_lbiases[i][0].unsqueeze(0).clone()}, topk_idx= i)
                milp_verr = milp_verifier.solv_MILP()
                if i in self.unverified_indices:
                    targetted_topk = int(max(milp_verr, torch.sum(ewf[0] >= 0).item()))
                    print(f"targetted topk {targetted_topk}")
                    print(f"abc topk {int(torch.sum(ewf[0] >= 0).item())}")
                    if targetted_topk <= 9 - topk and targetted_topk > 9 - topk -2:
                        indices = torch.tensor([i], device=self.device)
                        topk_loss = self.topk_strong_bounding(indices=indices, topk=topk)
                        topk_loss = torch.stack(topk_loss, dim=0)
                        print(f"\ntopk loss min {topk_loss.min()}")
                        if topk_loss.min() >= 0.0:
                            targetted_topk = max(targetted_topk, 10 - topk)   
                    self.property_result.meta_topk.append(targetted_topk)
                    self.property_result.abc_topk.append(int(torch.sum(ewf[0] >= 0).item()))
                    if self.property_result.meta_topk[-1] < self.property_result.abc_topk[-1]:
                        self.property_result.meta_topk[-1] = self.property_result.abc_topk[-1]
            else:
                print('Currently NOT Individual Verified ', self.unverified_indices)
                self.run_refinement_bab(torch.tensor([i] * len(target_lAs)),device = self.device, target = target_lAs, 
                                target_biases= target_lbiases, selected_lbiases = selected_lbiases, meta_timeout = self.meta_start + self.args.total_time)
            if time.time() - self.meta_start > self.args.total_time:
                return milp_tasks, selected_lAs, selected_lbiases
            if self.args.store_time_trace:
                milp_verifier = self.get_milp(selected_lAs, {k : [vv.detach().clone() for vv in v] for k, v in selected_lbiases.items()}, self.unverified_indices.clone(), ginf = self.args.gscale)
                cross_ex_verified_tuples = self.get_verified_tuples(max(self.args.execution_count_dct.keys()), self.cross_ex_loss, self.tuple_of_indices_cross_ex)
                milp_thread = threading.Thread(target = self.solv_async_milp, args= (milp_verifier, deepcopy(self.individual_verified), 'Targetted', cross_ex_verified_tuples, ))
                milp_thread.start()
                milp_tasks.append(milp_thread)
           
        self.property_result.alg_results['Individual Verified']['accuracy'] = (self.individual_verified)/(self.args.count_per_prop) * 100.0
        self.property_result.alg_results['Individual Verified']['time'] = time.time() - self.meta_start
        return milp_tasks, selected_lAs, selected_lbiases
    
    def bab_cross_ex_refinement(self, milp_tasks, selected_lAs, selected_lbiases, algorithm, max_iteration = 1e9):
        all_tuples, all_refined_tuples, orig_scores, final_indices, final_refined_indices = self.select_indices_branching2()
        timeout = False
        iteration = 0
        while len(all_tuples) > 0:
            verified = []
            unverified = []
            lenf = len(final_indices)
            assert lenf == 1
            numb = 0
            for (i, tmp1), (i, tmp2) in zip(final_indices, final_refined_indices):
                assert len(tmp1) % i == 0
                num_combs =  (len(tmp1) // i)
                assert num_combs == 1
                self.run_cross_executional_refinement(i, tmp1, tmp2, True, num_combs, self.meta_start + self.args.total_time)
                new_verified = []
                # pos_s, pos_e = (-lenf + numb), (-lenf + numb + num_combs)
                # if pos_e > 0:
                #     raise RuntimeError
                # elif pos_e == 0:
                #     tuples_done = self.tuple_of_indices_cross_ex_branching[i][pos_s:]
                # else:
                #     tuples_done = self.tuple_of_indices_cross_ex_branching[i][pos_s:pos_e]
                # try:
                #     assert len(tuples_done) == len(self.cross_ex_loss_branching[i][(-num_combs):])
                # except:
                #     import pdb;pdb.set_trace()
                tuples_done = self.tuple_of_indices_cross_ex_branching[i][-1:]
                for tup, loss in zip(tuples_done, self.cross_ex_loss_branching[i][(-num_combs):]):
                    if loss >= 0:
                        new_verified.append(tup)
                    else:
                        unverified.append(tup)
                verified.extend(new_verified)
                numb += 1
                if milp_tasks is not None and self.args.store_time_trace and len(new_verified) > 0:
                    milp_verifier = self.get_milp(selected_lAs, {k : [vv.detach().clone() for vv in v] for k, v in selected_lbiases.items()}, self.unverified_indices.clone(), ginf = self.args.gscale)
                    cross_ex_verified_tuples = self.get_verified_tuples(max(self.args.execution_count_dct.keys()), self.cross_ex_loss_branching, self.tuple_of_indices_cross_ex_branching)
                    milp_thread = threading.Thread(target = self.solv_async_milp, args= (milp_verifier, deepcopy(self.individual_verified), algorithm, cross_ex_verified_tuples, ))
                    milp_thread.start()
                    milp_tasks.append(milp_thread)

                if time.time() - self.meta_start > self.args.total_time:
                    timeout = True 
                    break
                
                iteration += 1
                if iteration >= max_iteration:
                    timeout = True 
                    break
            
            if timeout:
                break
            
            all_tuples, all_refined_tuples, orig_scores, final_indices, final_refined_indices = self.update_selection_branching(all_tuples, all_refined_tuples, orig_scores, verified, unverified)
        
        self.property_result.complete_tuples = {k : deepcopy(v) for k, v in self.tuple_of_indices_cross_ex_branching.items()}
        self.property_result.complete_losses = {k : torch.tensor(v) for k, v in self.cross_ex_loss_branching.items()}
        return milp_tasks

    def complete_relational_verifier(self, refinement_indices, alpha_crown_lb, racoon_milp):
        if racoon_milp is not None:
            milp_tasks = [threading.Thread(target = self.solv_async_milp, args = (racoon_milp, deepcopy(self.individual_verified), 'Initial CrossEx', ))] 
            milp_tasks[-1].start()
        else:
            milp_tasks = []

        assert self.args.verify_mode == 'verify_meta'
        milp_tasks, selected_lAs, selected_lbiases = self.targetted_refinement(refinement_indices, alpha_crown_lb, milp_tasks)
        time_diff = time.time() - self.meta_start
        if (time.time() - self.meta_start > self.args.total_time):
            for t in milp_tasks:
                t.join()
            return
        
        if not self.args.store_time_trace:
            cross_ex_verified_tuples = self.get_verified_tuples(max(self.args.execution_count_dct.keys()), self.cross_ex_loss, self.tuple_of_indices_cross_ex)
            milp_verifier = self.get_milp(selected_lAs, {k : [vv.detach().clone() for vv in v] for k, v in selected_lbiases.items()}, self.unverified_indices.clone(), ginf=self.args.gscale)
            
            milp_thread = threading.Thread(target = self.solv_async_milp, args= (milp_verifier, deepcopy(self.individual_verified), 'Targetted', cross_ex_verified_tuples, ))
            milp_thread.start()
            milp_tasks.append(milp_thread)
        
        if self.args.addl_prune > 0:
            cross_executional_indices_from_refinement = self.select_indices(lower_bound= alpha_crown_lb, threshold = None)
            cross_executional_indices = refinement_indices[cross_executional_indices_from_refinement].detach().cpu().numpy()
            unverified = set(self.unverified_indices.tolist())
            for i in cross_executional_indices[self.args.num_targetted: (self.args.num_targetted + self.args.addl_prune)].tolist():
                if (time.time() - self.meta_start > self.args.total_time):
                    for t in milp_tasks:
                        t.join()
                    return
                if (i not in unverified):
                    continue 
                is_verified = self.run_refinement_bab(torch.tensor([i]),device = self.device)
                if is_verified: 
                    unverified.remove(i)
                    self.individual_verified += 1
                    dt = time.time() - self.meta_start 
                    self.property_result.verified_dict[dt] = ((self.individual_verified)/(self.args.count_per_prop) * 100.0, 'Individual Verified')
            self.unverified_indices = torch.as_tensor(list(unverified), device = self.unverified_indices.device)
            cross_ex_verified_tuples = self.get_verified_tuples(max(self.args.execution_count_dct.keys()), self.cross_ex_loss, self.tuple_of_indices_cross_ex)
            milp_verifier = self.get_milp(selected_lAs, {k : [vv.detach().clone() for vv in v] for k, v in selected_lbiases.items()}, self.unverified_indices.clone(), ginf=self.args.gscale)
            milp_thread = threading.Thread(target = self.solv_async_milp, args= (milp_verifier, deepcopy(self.individual_verified), 'Strong Branch + Strong Bound', cross_ex_verified_tuples, ))
            milp_thread.start()
            milp_tasks.append(milp_thread)  
              
        milp_tasks = self.bab_cross_ex_refinement(milp_tasks, selected_lAs, selected_lbiases, 'Strong Branch + Strong Bound')
        if time.time() - self.meta_start < self.args.total_time:
            cross_ex_verified_tuples = self.get_verified_tuples(max(self.args.execution_count_dct.keys()), self.cross_ex_loss_branching, self.tuple_of_indices_cross_ex_branching)
            milp_verifier = self.get_milp(selected_lAs, {k : [vv.detach().clone() for vv in v] for k, v in selected_lbiases.items()}, self.unverified_indices.clone(), ginf = self.args.gscale)
            
            milp_thread = threading.Thread(target = self.solv_async_milp, args= (milp_verifier, deepcopy(self.individual_verified), 'Strong Branch + Strong Bound', cross_ex_verified_tuples, ))
            milp_thread.start()
            milp_tasks.append(milp_thread)
    
        for t in milp_tasks:
            t.join()

    
    def cross_ex_branching(self, milp_tasks, time_diff):
        selected_lAs, selected_lbiases = self.select_apprx('loss', self.args.max_linear_apprx)
        self.tuple_of_indices_cross_ex_branching = {}
        self.cross_ex_loss_branching = {}
        self.meta_start = time.time()
        self.args.total_time -= time_diff
        self.bab_cross_ex_refinement(None, selected_lAs, selected_lbiases, 'BaB+CrossEx')
        
        cross_ex_verified_tuples = self.get_verified_tuples(max(self.args.execution_count_dct.keys()), self.cross_ex_loss_branching, self.tuple_of_indices_cross_ex_branching)
        milp_verifier = self.get_milp(selected_lAs, {k : [vv.detach().clone() for vv in v] for k, v in selected_lbiases.items()}, self.unverified_indices.clone(), ginf = self.args.gscale)
        
        milp_thread = threading.Thread(target = self.solv_async_milp, args= (milp_verifier, deepcopy(self.individual_verified), 'BaB+CrossEx', cross_ex_verified_tuples, ))
        milp_thread.start()
        milp_tasks.append(milp_thread)
        return milp_tasks
        
        
        
        

    def verify_meta(self):
        torch.cuda.empty_cache()
        self.meta_start = time.time()
        assert len(self.nets) == 1
        if self.args.raven_mode not in [RavenMode.UAP, RavenMode.UAP_BINARY]:
            raise NotImplementedError(f'Currently {self.args.raven_mode} is not supported')
        
        arguments.Config.parse_no_config()
        arguments.Config['bab']['timeout'] = self.args.bab_timeout
        arguments.Config['solver']['batch_size'] = self.args.bab_batch_size
        arguments.Config['bab']['branching']['method'] = self.args.branching_method
        self.property_result.raven_args = self.args 
        self.initialize_models()
        
        if self.args.parallelize_executions:
            raise NotImplementedError       
            self.initialize_for_parallel()
                
        self.shift_to_device(self.device, self.bounded_models, self.prop)
        
        self.verify_crown()
        if self.individual_verified == self.args.count_per_prop:
            return

        refinement_indices, alpha_crown_lb = self.verify_alpha_crown()
        if self.individual_verified == self.args.count_per_prop:
            return 
        
        torch.cuda.empty_cache()
        if self.args.prune_before_refine > 0:
            self.run_abc(self.args.prune_before_refine)
            refinement_indices_lst = refinement_indices.tolist()
            lbs_idx = [refinement_indices_lst.index(i) for i in self.unverified_indices.tolist()]
            alpha_crown_lb = alpha_crown_lb[lbs_idx]
            refinement_indices = self.unverified_indices

        if len(self.unverified_indices) > 1: 
            racoon_milp = self.run_racoon(refinement_indices, alpha_crown_lb)
        else:
            racoon_milp = None
        
        self.complete_relational_verifier(refinement_indices, alpha_crown_lb, racoon_milp)
    

    
    def verify_topk(self):
        torch.cuda.empty_cache()
        self.meta_start = time.time()
        assert len(self.nets) == 1
        if self.args.raven_mode not in [RavenMode.UAP, RavenMode.UAP_BINARY]:
            raise NotImplementedError(f'Currently {self.args.raven_mode} is not supported')
        
        arguments.Config.parse_no_config()
        arguments.Config['bab']['timeout'] = self.args.bab_timeout
        arguments.Config['solver']['batch_size'] = self.args.bab_batch_size
        
        self.property_result.raven_args = self.args 
        self.initialize_models()
        
        self.args.max_linear_apprx = 1
        self.args.max_targetted = 1
        
        
        if self.args.parallelize_executions:
            raise NotImplementedError       
            self.initialize_for_parallel()
                
        self.shift_to_device(self.device, self.bounded_models, self.prop)
        
        self.verify_crown()
        if self.individual_verified == self.args.count_per_prop:
            return

        refinement_indices, alpha_crown_lb = self.verify_alpha_crown(topk = 2)
        if self.individual_verified == self.args.count_per_prop:
            return
        
        self.args.num_targetted = self.args.count_per_prop - self.individual_verified 

        self.targetted_refinement(refinement_indices, alpha_crown_lb, None)

    def verify_abc(self):
        torch.cuda.empty_cache()
        self.meta_start = time.time()
        assert len(self.nets) == 1
        if self.args.raven_mode not in [RavenMode.UAP, RavenMode.UAP_BINARY]:
            raise NotImplementedError(f'Currently {self.args.raven_mode} is not supported')
        
        arguments.Config.parse_no_config()
        if self.args.gcp_crown:
            arguments.Config['bab']['cut']['enabled'] = True
            arguments.Config['bab']['cut']['cplex_cuts'] = True
            arguments.Config['bab']['cut']['bab_cut'] = True
        arguments.Config['bab']['timeout'] = self.args.bab_timeout
        arguments.Config['solver']['batch_size'] = self.args.bab_batch_size
        arguments.Config['bab']['branching']['method'] = self.args.branching_method
        self.property_result.raven_args = self.args 
        
        self.initialize_models()
        self.shift_to_device(self.device, self.bounded_models, self.prop)
        self.verify_alpha_crown()
        self.run_abc()
        self.property_result.unverified_indices = self.unverified_indices.tolist()
        return

