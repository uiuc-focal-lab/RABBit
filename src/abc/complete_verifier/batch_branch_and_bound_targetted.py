#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for activation space split."""
import time
import random
import numpy as np
import torch
import copy
from collections import defaultdict

from auto_LiRPA.utils import stop_criterion_sum, stop_criterion_batch_any, stop_criterion_batch_topk, stop_criterion_min
from branching_domains import merge_domains_params, SortedReLUDomainList, BatchedReLUDomainList
from branching_heuristics import choose_node_parallel_FSB, choose_node_parallel_crown, choose_node_parallel_kFSB, choose_node_parallel_targetted
from functools import partial
import arguments
from specification_tree import TreeNode, ProofPQ, SpecificationTree, PQNode
from branching_domains import select_batch
from adv_domains import AdvExamplePool
from bab_attack import beam_mip_attack, find_promising_domains, bab_attack
from cut_utils import fetch_cut_from_cplex, generate_cplex_cuts, clean_net_mps_process, cplex_update_general_beta
from bab_bounds_multiple import update_bounds_parallel_multiple
from src.gurobi_certifier import RavenLPtransformer

Visited, Flag_first_split = 0, True
Use_optimized_split = False
all_node_split = False
total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0


def build_history(history, split, orig_lbs, orig_ubs):
    """
    Generate fake history and fake lower and upper bounds for new domains
    history: [num_domain], history of the input domains
    split: [num_copy * num_domain], split decision for each new domain.
    orig_lbs, orig_ubs: [num_relu_layer, num_copy, num_domain, relu_layer.shape]
    """
    new_history = []
    num_domain = len(history)
    num_split = len(split)//num_domain

    num_layer = len(orig_lbs)

    def generate_history(heads, splits, orig_lbs, orig_ubs, domain_idx):
        '''
        Generate [num_copy] fake history and fake lower and upper bounds for an input domain.
        '''
        for pos in range(num_split-1):
            num_history = len(heads)
            for i in range(num_history):
                decision_layer = splits[pos*num_domain+domain_idx][0][0]
                decision_index = splits[pos*num_domain+domain_idx][0][1]

                for l in range(num_layer):
                    orig_ubs[l][num_history+i][domain_idx] = orig_ubs[l][i][domain_idx]
                    orig_lbs[l][num_history+i][domain_idx] = orig_lbs[l][i][domain_idx]

                orig_lbs[decision_layer][i][domain_idx].view(-1)[decision_index] = 0.0
                heads[i][decision_layer][0].append(decision_index)
                heads[i][decision_layer][1].append(1.0)
                heads.append(copy.deepcopy(heads[i]))
                orig_ubs[decision_layer][num_history+i][domain_idx].view(-1)[decision_index] = 0.0
                heads[-1][decision_layer][1][-1] = -1.0
        return heads
    new_history_list = []
    for i in range(num_domain):
        new_history_list.append(generate_history([history[i]], split, orig_lbs, orig_ubs, i))

    for i in range(len(new_history_list[0])):
        for j in range(num_domain):
            new_history.append(new_history_list[j][i])
    # num_copy * num_domain
    return new_history, orig_lbs, orig_ubs



def get_from_store(store, key):
    return [s[key] for s in store]

def are_equal(list1, list2):
    return all(torch.equal(tensor1, tensor2) for tensor1, tensor2 in zip(list1, list2))

def get_verified_count(lower_bnd):
    return torch.sum(lower_bnd.detach().cpu().min(axis=1)[0] > 0).numpy() if isinstance(lower_bnd, torch.Tensor) else sum([(lb.detach().cpu().min() > 0).item() for lb in lower_bnd])


def pick_image_to_split(ds, net, batch, pre_relu_indices, branching_reduceop, branching_candidates):
    #go over and pick batch domains from the Domains for each image
    domains_store = []
    branching_decision, split_depth, branch_idx = None, None, -1
    
    for d in ds:
        domains_params = d.pick_out(batch=batch, device=net.x.device, ret_tree_node = True)
        mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs, rhs, tree_nodes, targetted_loss = domains_params
        domains_store.append(dict(mask = mask, lAs = lAs, orig_lbs = orig_lbs, orig_ubs = orig_ubs, slopes = slopes, betas = betas, intermediate_betas = intermediate_betas, selected_domains = selected_domains, cs = cs, rhs = rhs, tree_nodes = tree_nodes, targetted_loss = targetted_loss))


    #only perform splitting if the mask of any image is not None
    if any([bool(dstore['mask'] is not None) for dstore in domains_store]):
        for store in domains_store:
            selected_domains = store['selected_domains']

            history = [sd.history for sd in selected_domains]
            split_history = [sd.split_history for sd in selected_domains]
            

            # Here we check the length of current domain list.
            # If the domain list is small, we can split more layers.
            min_batch_size = min(arguments.Config["solver"]["min_batch_size_ratio"]*arguments.Config["solver"]["batch_size"], batch)

            if store['orig_lbs'][0].shape[0] < min_batch_size:
                # Split multiple levels, to obtain at least min_batch_size domains in this batch.
                split_depth = int(np.log(min_batch_size)/np.log(2))

                if store['orig_lbs'][0].shape[0] > 0:
                    split_depth = max(int(np.log(min_batch_size/store['orig_lbs'][0].shape[0])/np.log(2)), 0)
                split_depth = max(split_depth, 1)
            else:
                split_depth = 1

            print("batch: ", store['orig_lbs'][0].shape, "pre split depth: ", split_depth)
            # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
            branching_candidates = max(branching_candidates, split_depth)
            store['branching_candidates'] = branching_candidates
            store['selected_domains'] = selected_domains
            store['history'] = history
            store['split_depth'] = split_depth
            store['split_history'] = split_history

        getter = partial(get_from_store, domains_store)
        branching_decision, split_depth, branch_idx = choose_node_parallel_targetted(getter('orig_lbs'), getter('orig_ubs'), getter('mask'), net, pre_relu_indices, getter('lAs'),
                                           batch = batch, branching_reduceop=branching_reduceop,split_depth= getter('split_depth'),
                                              cs= getter('cs'), rhs= getter('rhs'))
        # branching_decision, split_depth = choose_node_parallel_crown(domains_store[0]['orig_lbs'], domains_store[0]['orig_ubs'], domains_store[0]['mask'], net, pre_relu_indices, domains_store[0]['lAs'], 
        #                                                              batch=batch, branching_reduceop=branching_reduceop, split_depth=domains_store[0]['split_depth'], cs= domains_store[0]['cs'], rhs=domains_store[0]['rhs'])
    
    return domains_store, branching_decision, split_depth, 0


def batch_verification3(ds, net, batch, pre_relu_indices, growth_rate, fix_intermediate_layer_bounds=True,
                    stop_func=stop_criterion_sum, multi_spec_keep_func=lambda x: torch.all(x, dim=-1), 
                    target_lAs = None, target_biases = None):
    
    global Visited, Flag_first_split
    global Use_optimized_split
    global total_pickout_time, total_decision_time, total_solve_time, total_add_time

    branching_reduceop = arguments.Config['bab']['branching']['reduceop']
    branching_candidates = arguments.Config["bab"]["branching"]["candidates"]

    domains_store, branching_decision, split_depth, branch_idx = pick_image_to_split(ds, net, batch, pre_relu_indices, branching_reduceop, branching_candidates) 
    
    #all images are fully split so don't perform branching
    if branching_decision is None:
        return True
    
    else:
        print("batch: ", domains_store[branch_idx]['orig_lbs'][0].shape, "post split depth: ", split_depth)

        print('splitting decisions: ')
        for l in range(split_depth):
            print("split level {}".format(l), end=": ")
            for b in range(min(10, len(domains_store[branch_idx]['history']))):
                print(branching_decision[l*len(domains_store[branch_idx]['history']) + b], end=" ")
            print('')
        # print the first two split for first 10 domains.

        if not Use_optimized_split:
            split = {}
            split_copy = {}
            # split["decision"]: selected domains (next batch/2)->node list->node: [layer, idx]
            split["decision"] = [[bd] for bd in branching_decision]
            split_copy['decision'] = [[bd] for bd in branching_decision]
            # split["split"]: selected domains (next batch/2)->node list->float coefficients
            split["coeffs"] = [[1.] for i in range(len(branching_decision))]

        num_copy = (2**(split_depth-1))
        num_domain = len(domains_store[branch_idx]['history'])
        if num_copy > 1:
            for idx in range(len(ds)):
                orig_lbs = [lb.unsqueeze(0).repeat(num_copy, *[1]*len(lb.shape)) for lb in domains_store[idx]['orig_lbs']]
                orig_ubs = [ub.unsqueeze(0).repeat(num_copy, *[1]*len(ub.shape)) for ub in domains_store[idx]['orig_ubs']]
                # 4 * [num_copy, num_domain, xxx]

                assert num_domain == len(domains_store[idx]['history'])

                # create fake history for each branch.
                # TODO: set origlbs and orig_ubs

                domains_store[idx]['history'], orig_lbs, orig_ubs = build_history(domains_store[idx]['history'], split_copy['decision'], orig_lbs, orig_ubs)

                
                # set the slopes for each branch
                for k, v in domains_store[idx]['slopes'].items():
                    for kk, vv in v.items():
                        v[kk] = torch.cat([vv] * num_copy, dim=2)

                # create fake split_history for each branch.
                domains_store[idx]['split_history'] = domains_store[idx]['split_history'] * num_copy

                # cs needs to repeat
                domains_store[idx]['cs'] = torch.cat([domains_store[idx]['cs']] * num_copy, dim=0)
                domains_store[idx]['targetted_loss'] = torch.cat([domains_store[idx]['targetted_loss']] * num_copy, dim=0)

                new_betas = []
                new_intermediate_betas = []
                for i in range(num_copy):
                    for j in range(len(domains_store[idx]['betas'])):
                        new_betas.append(domains_store[idx]['betas'][j])
                        new_intermediate_betas.append(domains_store[idx]['intermediate_betas'][j])
                
                domains_store[idx]['betas'] = new_betas
                domains_store[idx]['intermediate_betas'] = new_intermediate_betas

                domains_store[idx]['orig_lbs'] = [lb.view(-1, *lb.shape[2:]) for lb in orig_lbs]
                domains_store[idx]['orig_ubs'] = [ub.view(-1, *ub.shape[2:]) for ub in orig_ubs]
                
                
                domains_store[idx]['rhs'] = torch.cat([domains_store[idx]['rhs']] * num_copy, dim=0)
            
                        # create split for num_copy * num_domain.
            # we only keep the last split since the first few ones has been split with build_history
            split['decision'] = split['decision'][-num_domain:] * num_copy
            split['coeffs'] = split['coeffs'][-num_domain:] * num_copy
            branching_decision = branching_decision[-num_domain:] * num_copy
        

        getter = partial(get_from_store, domains_store)
        solve_time = time.time()
        history = domains_store[branch_idx]['history']
        
        for hist in getter('history'):
            assert hist == history
        
        ret = net.update_bounds_parallel_targetted(split, history, fix_intermediate_layer_bounds=fix_intermediate_layer_bounds, decision_thresh= domains_store[branch_idx]['rhs'], stop_criterion_func = stop_func(torch.cat([domains_store[branch_idx]['rhs']] * 2 * len(ds))), multi_spec_keep_func=multi_spec_keep_func, all_img_pre_lb = getter('orig_lbs'), all_img_pre_ub = getter('orig_ubs'), all_img_betas = getter('betas'), all_img_slopes = getter('slopes'), all_img_cs = getter('cs'), target_lAs = target_lAs, target_biases = target_biases, all_img_targetted_loss = getter('targetted_loss'))

        if arguments.Config["cross_ex"]["enable_cross_ex"]:
            dom_ub, dom_lb, dom_ub_point, dom_lb_all, dom_ub_all, lAs, slopes, split_history, betas, intermediate_betas, primals, dom_cs, A = ret
        else:
            dom_ub, dom_lb, dom_ub_point, dom_lb_all, dom_ub_all, lAs, slopes, split_history, betas, intermediate_betas, primals, dom_cs = ret
        
        print('subproblem lb: ',  torch.min(dom_lb))
        solve_time = time.time() - solve_time
        total_solve_time += solve_time
        add_time = time.time()
        batch = len(branching_decision)
        # If intermediate layers are not refined or updated, we do not need to check infeasibility when adding new domains.
        check_infeasibility = not (fix_intermediate_layer_bounds)
        selected_domains = domains_store[branch_idx]['selected_domains']
        per_domain_losses = net.cross_refinement_results['targetted_loss'].reshape(len(domains_store), -1, *net.cross_refinement_results['targetted_loss'].shape[1:]).detach().cpu() 
        
        per_child_losses = [dl.squeeze(1) for dl in torch.chunk(per_domain_losses, per_domain_losses.shape[1], dim = 1)]
        new_tree_nodes = [TreeNode(targetted_loss= per_child_losses[i]) for i in range(len(A))]
        for i, node in enumerate(new_tree_nodes):
            domains_store[branch_idx]['tree_nodes'][i % len(selected_domains)].add_child(node)
        dom_cs = dom_cs.cpu()
        left_indexer = torch.nonzero(torch.min(dom_lb[0][:batch], dim = 1).values < 0, as_tuple = False).view(-1)
        right_indexer =  torch.nonzero(torch.min(dom_lb[0][batch:(2*batch)], dim = 1).values < 0, as_tuple = False).view(-1)
        stop = False
        for i in range(len(ds)):
            add_time = time.time()
            depths = [domain.depth + split_depth - 1 for domain in selected_domains] * num_copy * 2
            repeated_split = ds[i].add(lAs[i], dom_lb[i], dom_ub[i], dom_lb_all[i], dom_ub_all[i], domains_store[i]['history'], depths, slopes[i], betas[i], 
                    split_history[i], branching_decision, domains_store[i]['rhs'], intermediate_betas[i], check_infeasibility, dom_cs[i], (2*num_copy)*batch, new_tree_nodes, left_indexer, right_indexer, per_domain_losses[i])
            dt = time.time() - add_time
            stop = (stop or repeated_split)
            try:
                assert dt <= 2.0
            except:
                pass
        return stop


def relu_bab_parallel_targetted(net, domains, xs, use_neuron_set_strategy=False, refined_lower_bounds=None,
                      refined_upper_bounds=None, activation_opt_params=None,
                      reference_slopes=None, reference_lA=None, attack_images=None,
                      timeout=None, refined_betas=None, rhs=0,
                      target_lAs = None, target_biases=None):
    # the crown_lower/upper_bounds are present for initializing the unstable indx when constructing bounded module
    # it is ok to not pass them here, but then we need to go through a CROWN process again which is slightly slower


    start = time.time()
    assert arguments.Config["cross_ex"]["enable_cross_ex"]
    #turn off pruning in iteration when running cross executional refinement
    if arguments.Config["cross_ex"]["enable_cross_ex"]:
        arguments.Config["bab"]["pruning_in_iteration"] = False
    
    # All supported arguments.
    global Visited, Flag_first_split, all_node_split 
    global total_pickout_time, total_decision_time, total_solve_time, total_add_time

    all_node_split = [False] * xs.shape[0]
    total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0

    timeout = timeout or arguments.Config["bab"]["timeout"]
    max_domains = arguments.Config["bab"]["max_domains"]
    batch = arguments.Config["solver"]["batch_size"]
    record = arguments.Config["general"]["record_bounds"]
    opt_intermediate_beta = False
    lp_test = arguments.Config["debug"]["lp_test"]
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    use_bab_attack = arguments.Config["bab"]["attack"]["enabled"]
    max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
    min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
    cut_enabled = arguments.Config["bab"]["cut"]["enabled"]
    lp_cut_enabled = arguments.Config["bab"]["cut"]["lp_cut"]
    use_batched_domain = arguments.Config["bab"]["batched_domain_list"]
    
    
    if not isinstance(rhs, torch.Tensor):
        rhs = torch.tensor(1e9)
    decision_thresh = rhs.expand(xs.shape[0], 1).to(net.device)

    # general (multi-bounds) output for one C matrix
    # any spec >= rhs, then this sample can be stopped; if all samples can be stopped, stop = True, o.w., False
    stop_criterion = stop_criterion_min if arguments.Config["cross_ex"]["enable_cross_ex"] else stop_criterion_batch_any
    multi_spec_keep_func = lambda x: torch.all(x, dim=-1)

    Visited, Flag_first_split, global_ub = [0] * xs.shape[0], [True] * xs.shape[0], [np.inf] * xs.shape[0]
    betas = None
    if arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]:
        if arguments.Config["cross_ex"]["enable_cross_ex"]:
            global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, betas, As = net.build_the_model_with_refined_bounds(
            domains, xs, None, None, stop_criterion_func=stop_criterion(decision_thresh), reference_slopes=None,
            cutter=net.cutter, target_lAs = target_lAs)
            logit_lbs = global_lb.clone()

            
    elif refined_lower_bounds is None or refined_upper_bounds is None:
        assert arguments.Config["general"]["enable_incomplete_verification"] is False
        if arguments.Config["cross_ex"]["enable_cross_ex"]:
            global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, attack_image, As = net.build_the_model(
            domains, xs, stop_criterion_func=stop_criterion(decision_thresh), target_lAs = target_lAs, target_biases = target_biases)

            
    else:
        if arguments.Config["cross_ex"]["enable_cross_ex"]:
            global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, betas, As = net.build_the_model_with_refined_bounds(
            domains, xs, refined_lower_bounds, refined_upper_bounds, activation_opt_params, reference_lA=reference_lA,
            stop_criterion_func=stop_criterion(decision_thresh), reference_slopes=reference_slopes,
            cutter=net.cutter, refined_betas=refined_betas, target_lAs = target_lAs)

        # release some storage to save memory
        if activation_opt_params is not None: del activation_opt_params
        torch.cuda.empty_cache()
    
    targetted_loss= net.cross_refinement_results['targetted_loss'].detach().cpu()

    if stop_criterion(decision_thresh)(global_lb[0:1]).all():
        if arguments.Config["cross_ex"]["enable_cross_ex"]:
            return True, SpecificationTree(root = TreeNode(targetted_loss= targetted_loss))
       
    if not opt_intermediate_beta:
        # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
        # We only keep the alpha for the last layer.
        if not arguments.Config['solver']['beta-crown'].get('enable_opt_interm_bounds', False):
            # new_slope shape: [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}] for each sample in batch]
            new_slope = {}
            kept_layer_names = [net.net.final_name]
            kept_layer_names.extend(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
            print(f'Keeping slopes for these layers: {kept_layer_names}')
            for relu_layer, alphas in slope.items():
                new_slope[relu_layer] = {}
                for layer_name in kept_layer_names:
                    if layer_name in alphas:
                        new_slope[relu_layer][layer_name] = alphas[layer_name]
                    else:
                        print(f'Layer {relu_layer} missing slope for start node {layer_name}')
        else:
            new_slope = slope
        
        
    if use_batched_domain:
        assert not use_bab_attack, "Please disable batched_domain_list to run BaB-Attack."
        DomainClass = BatchedReLUDomainList
    else:
        DomainClass = SortedReLUDomainList
    
    # This is the first (initial) domain.
    num_initial_domains = net.c.shape[0]
    per_img_domains = []
    
    specification_tree = SpecificationTree(root = TreeNode(targetted_loss= targetted_loss))
    #we have to seperate the domains per image for bab across multiple images
    for i in range(num_initial_domains):
        per_img_domains.append(DomainClass([la[i:i+1] for la in lA], global_lb[i:i+1], global_ub[i:i+1], [lbds[i:i+1] for lbds in lower_bounds], 
                                           [ubds[i:i+1] for ubds in upper_bounds], {k : {kk : vv[:, :, i:i+1, :] for kk, vv in v.items()} for k, v in new_slope.items()}, 
                                           copy.deepcopy(history), [0], net.c[i:i+1], decision_thresh[i][0].cpu(), betas, 1, interm_transfer=arguments.Config["bab"]["interm_transfer"], tree_node= specification_tree.root, loss= targetted_loss[i:i+1]))
            

    if not arguments.Config["bab"]["interm_transfer"]:
        # tell the AutoLiRPA class not to transfer intermediate bounds to save time
        net.interm_transfer = arguments.Config["bab"]["interm_transfer"]
    

    tot_ambi_nodes = 0
    # only pick the first copy from possible multiple x
    updated_mask = [mask[0:1] for mask in updated_mask]
    for i, layer_mask in enumerate(updated_mask):
        n_unstable = int(torch.sum(layer_mask).item())
        print(f'layer {i} size {layer_mask.shape[1:]} unstable {n_unstable}')
        tot_ambi_nodes += n_unstable

    print(f'-----------------\n# of unstable neurons for Network: {tot_ambi_nodes}\n-----------------\n')
    net.tot_ambi_nodes = tot_ambi_nodes

    iter = 0
    assert all([len(d) == len(per_img_domains[0]) for d in per_img_domains])
    
    while all([len(d) > 0 for d in per_img_domains]):
        assert all([len(d) == len(per_img_domains[0]) for d in per_img_domains])
        if any([bool(len(domain) > 80000 and len(domain) % 10000 < batch * 2 and use_neuron_set_strategy) for domain in per_img_domains]):
            # neuron set  bounds cost more memory, we set a smaller batch here
            stop = batch_verification3(per_img_domains, use_neuron_set_strategy, int(batch/2), pre_relu_indices, 0,
                                    fix_intermediate_layer_bounds=False, stop_func= stop_criterion,
                                    multi_spec_keep_func=multi_spec_keep_func,  target_lAs = target_lAs, target_biases = target_biases)
        else:
            stop = batch_verification3(per_img_domains, net, batch, pre_relu_indices, 0,
                                    fix_intermediate_layer_bounds=not opt_intermediate_beta,
                                    stop_func= stop_criterion, multi_spec_keep_func=multi_spec_keep_func, target_lAs = target_lAs, target_biases = target_biases)
        
        if stop:
            break
        
        iter += 1
        if (time.time() - start > timeout):
            print("TIMEOUT!")
            break
    
    is_verified = all([len(d) == 0 for d in per_img_domains])
    
    clean_net_mps_process(net)
    for d in per_img_domains:
        del d

    return is_verified, specification_tree