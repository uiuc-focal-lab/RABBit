"""
A generic approach for the BnB based complete verification.
"""
from queue import PriorityQueue
import torch
import src.specs.spec as specs
import src.util as util
import src.parse as parse
from src.util import get_linear_layers
import time

from raven.src import config
from src.bnb import Split, is_relu_split
from src.bnb.proof_tree import ProofTree
from src.domains.deepz import ZonoTransformer
from src.common import Status
from src.util import compute_2dlist_intersection
from src.proof_transfer.pt_types import ProofTransferMethod, PRUNE, REORDERING
from multiprocessing import Pool


class BnB:
    def __init__(self, net, transformer, init_prop, split, template_store, adv_label=None, timeout=30, args=None, pruning_args=None):
        self.net = net
        self.transformer = transformer
        self.init_prop = init_prop
        self.split = split
        self.template_store = template_store
        self.adv_label = adv_label
        self.timeout = timeout
        self.args = args
        self.depth = 1
        self.init_time = time.time()
        self.global_status = Status.UNKNOWN
        self.global_lower_bound = 1e7
        self.perturbation_bounds = []
        self.verified_spec_queue = PriorityQueue()
        self.active_relu_list = []
        self.inactive_relu_list = []
        self.sparsification_result = None
        self.active_relu_list_computed = False
        self.inactive_relu_list_computed = False
        self.pruning_args=pruning_args
        self.pruning_enabled = None
        if self.pruning_args is not None:
            self.pruning_enabled = True
        else:
            self.pruning_enabled = False
        self.desired_perturbation_bound = 2.0
        self.pruning_layer = -1
        if self.pruning_enabled:
            if self.pruning_args.desired_perturbation is not None:
                self.desired_perturbation_bound = self.pruning_args.desired_perturbation
            if self.pruning_args.layers_to_prune is not None and len(self.pruning_args.layers_to_prune) > 0:
                self.pruning_layer = self.pruning_args.layers_to_prune[0]
        self.prev_pruning_layer = self.pruning_layer -1

        # Store proof tree for the BnB
        self.inp_template = self.template_store.get_template(self.init_prop)
        self.root_spec = None
        self.proof_tree = None

        self.cur_specs = self.get_init_specs(adv_label, init_prop)
        self.tree_size = len(self.cur_specs)
        self.prev_lb = None
        self.cur_lb = None

        # is feature layer bound computed 
        self.feature_layer_bound_computed = False
        self.feature_layer_lb = None
        self.feature_layer_ub = None
        self.adv_feature_layer_lb = None
        self.adv_feature_layer_ub = None

    def get_feature_layer_bound(self):
        if self.feature_layer_bound_computed:
            return self.feature_layer_lb, self.feature_layer_ub
        temp_sorted_speclist = PriorityQueue()
        curr_len = 0
        while not self.verified_spec_queue.empty():
            val, spec = self.verified_spec_queue.get()
            temp_sorted_speclist.put((val, spec))
            f_lb, f_ub = spec.get_feature_bounds()
            if curr_len == 0:
                self.feature_layer_lb = f_lb
                self.feature_layer_ub = f_ub
            else:
                self.feature_layer_lb = torch.minimum(self.feature_layer_lb, f_lb)
                self.feature_layer_ub = torch.maximum(self.feature_layer_ub, f_ub) 
            curr_len += 1
        self.verified_spec_queue = temp_sorted_speclist
        self.feature_layer_bound_computed = True
        # print("Feature layer bounds", self.feature_layer_lb, self.feature_layer_ub)
        return self.feature_layer_lb, self.feature_layer_ub


    def get_adv_feature_bounds(self):
        return self.adv_feature_layer_lb, self.adv_feature_layer_ub

    def get_active_relu_list(self):
        if self.global_status != Status.VERIFIED or self.verified_spec_queue.empty():
            return None
        return self.compute_active_relu_list()

    def get_scaling_factor_for_perturbation(self):
        if self.pruning_layer not in [-1, -2]:
            raise ValueError("Currently pruning supported in last two layers")
        return self.transformer.get_perturbation_scaling(self.pruning_layer)
            
    def compute_active_relu_list(self):
        if self.active_relu_list_computed:
            return self.active_relu_list
        temp_sorted_speclist = PriorityQueue()
        current_list = None
        while not self.verified_spec_queue.empty():
            val, spec = self.verified_spec_queue.get()
            temp_sorted_speclist.put((val, spec))
            current_list = compute_2dlist_intersection(current_list, spec.active_relus)
        self.active_relu_list = current_list
        self.verified_spec_queue = temp_sorted_speclist
        self.active_relu_list_computed = True
        return self.active_relu_list

    def get_inactive_relu_list(self):
        if self.global_status != Status.VERIFIED or self.verified_spec_queue.empty():
            return None
        return self.compute_inactive_relu_list()
    
    def compute_inactive_relu_list(self):
        if self.inactive_relu_list_computed:
            return self.inactive_relu_list
        temp_sorted_speclist = PriorityQueue()
        current_list = None
        while not self.verified_spec_queue.empty():
            val, spec = self.verified_spec_queue.get()
            temp_sorted_speclist.put((val, spec))
            current_list = compute_2dlist_intersection(current_list, spec.inactive_relus)
        self.inactive_relu_list = current_list
        self.verified_spec_queue = temp_sorted_speclist
        self.inactive_relu_list_computed = True
        return self.inactive_relu_list

    def get_final_perturbation_bound(self):
        if self.global_status != Status.VERIFIED or self.verified_spec_queue.empty():
            return None
        else:
            temp = self.verified_spec_queue.get()
            self.verified_spec_queue.put(temp)
            return temp[0]

    def get_init_specs(self, adv_label, init_prop):
        tree_avail = self.template_store.is_tree_available(init_prop)

        if tree_avail and type(self.args.pt_method) == PRUNE:
            proof_tree = self.template_store.get_proof_tree(init_prop)
            cur_specs = proof_tree.get_pruned_leaves(self.args.pt_method.threshold, self.split)
        elif tree_avail and self.args.pt_method == ProofTransferMethod.REUSE:
            proof_tree = self.template_store.get_proof_tree(init_prop)
            cur_specs = proof_tree.get_leaves()
        else:
            unstable_relus = self.get_unstable_relus(adv_label, init_prop)
            cur_specs = self.create_initial_specs(init_prop, unstable_relus)
        if tree_avail and self.root_spec is None:
            self.root_spec = self.template_store.get_proof_tree(init_prop).root
        return cur_specs

    def get_unstable_relus(self, adv_label, init_prop):
        status, unstable_relus, lb = self.verify_node(self.transformer, init_prop, adv_label=adv_label)
        if status != Status.UNKNOWN:
            self.global_status = status
            eta_norm = None
            if self.global_status == Status.VERIFIED:
                eta_norm = self.compute_eta_norm(self.prev_pruning_layer)
                if eta_norm is not None and lb >= 0:
                    self.perturbation_bounds.append(lb / eta_norm)
        return unstable_relus

    def get_sparsification_result(self):
        return self.sparsification_result

    # verify a particular specification.
    def verify_spec(self, spec, debug_info=False):
        self.update_transformer(spec.input_spec, relu_spec=spec.relu_spec)

        # Transformer is updated with new mask
        status, unstable_relus, lb = self.verify_node(self.transformer, spec.input_spec,
                                                        adv_label=self.adv_label)
       
        eta_norm = None
        # Compute the penaltimate layer norm as eta norm
        if lb is not None and lb >= 0:
            eta_norm = self.compute_eta_norm(self.prev_pruning_layer, debug_info=False)
            f_lb, f_ub = self.transformer.get_layer_bound(self.prev_pruning_layer)
            spec.update_feature_bounds(f_lb, f_ub)
        
        self.update_cur_lb(lb)
        spec.update_status(status, lb, eta_norm=eta_norm, 
                    active_relus=self.transformer.get_active_relu_list(), inactive_relus=self.transformer.get_inactive_relu_list())
        return status, lb        

    def run(self):
        """
        It is the public method called from the analyzer. @param split is a string that chooses the mode for relu splitting.
        Currently, the possible modes are (WIP):
        1) ReLU ESIP 2) Input grad
        """
        if self.global_status != Status.UNKNOWN:
            if self.global_status != Status.VERIFIED or not self.pruning_enabled:    
                return
            else:
                # Though the problem is verified that does guarantee that the bound is
                # good enough for sparsification of the network. Hence we should continue
                # further with splitting the nodes.
                self.global_status = Status.UNKNOWN
                print("Length of current specifications", len(self.cur_specs))
        
        split_score = self.set_split_score(self.init_prop, self.cur_specs, inp_template=self.inp_template)
        # d = True
        c = 0
        while self.continue_search():
            self.update_depth()
            c += 1
            self.prev_lb = self.cur_lb
            self.reset_cur_lb()

            for spec in self.cur_specs:
                self.update_transformer(spec.input_spec, relu_spec=spec.relu_spec)
                # Transformer is updated with new mask
                status, lb = self.verify_spec(spec=spec)
                # print("Improved bound", spec.get_perturbation_bound())
                if lb is not None and lb >= 0.0:
                    self.global_lower_bound = min(self.global_lower_bound, lb)
                
                if c == 1 and status != Status.VERIFIED:
                    f_lb, f_ub = spec.get_feature_bounds()
                    self.adv_feature_layer_lb = f_lb
                    self.adv_feature_layer_ub = f_ub

                if status == Status.ADV_EXAMPLE:
                    self.global_status = status
                    self.store_final_tree()
                    return

                if self.is_timeout():                    
                    self.store_final_tree()
                    return

            print('Current Lower bound:', self.cur_lb)

            # Each spec should hold the prev lb and current lb
            self.cur_specs, verified_specs = self.cur_specs.prune(self.split, split_score=split_score,
                                                                  inp_template=self.inp_template,
                                                                  args=self.args,
                                                                  net=self.net, perturbation_bound=None)
            
            if self.cur_specs is None or verified_specs is None:
                self.global_status = Status.UNKNOWN
                self.store_final_tree()
                return
            
            
            if len(self.cur_specs) > 0:
                self.global_status = Status.UNKNOWN
                self.store_final_tree()
                return
                                
            self.populate_verified_spec_queue(verified_specs=verified_specs)
            # self.update_perturbation_bounds(verified_specs=verified_specs)
           
            # Update the tree size
            self.tree_size += len(self.cur_specs)

        self.check_verified_status()
        # if self.global_status == Status.VERIFIED and self.pruning_enabled:
        #     self.improve_perturbation_bound(split_score=split_score)
        if self.global_status == Status.VERIFIED and self.args.enable_sparsification == True:
            linear_layers = get_linear_layers(self.net)
            self.sparsification_result = self.transformer.compute_sparsification(linear_layers[-1], self.adv_label, True)
            # if self.sparsification_result is not None:    
            #     print(self.sparsification_result)
        self.store_final_tree()


    def populate_verified_spec_queue(self, verified_specs):
        for spec in verified_specs:
            if spec.get_perturbation_bound() is not None:
                temp = (spec.get_perturbation_bound(), spec)
                self.verified_spec_queue.put(temp)
            else:
                raise ValueError("verified spec has none perturbation error")
    
    def improve_perturbation_bound(self, split_score=None):
        tolerence_lim = 1e-6
        while not self.verified_spec_queue.empty():
            if self.is_timeout():
                return
            curr_bound, spec = self.verified_spec_queue.get()
            print("Current perturbation bound ", curr_bound)
            # print("Curr bound", curr_bound)
            if self.desired_perturbation_bound is not None:
                if curr_bound > self.desired_perturbation_bound:
                    self.verified_spec_queue.put((curr_bound, spec))
                    break
            new_spec_list = spec.split_spec(self.split, split_score=split_score,
                                           inp_template=self.inp_template,
                                           args=self.args, net=self.net)
            # If specification splitting fails restore the existing set of specification
            # and return without splitting the problem into subproblems.
            if new_spec_list is None:
                print("Splitted spec list is none")
                self.verified_spec_queue.put((curr_bound, spec))
                return
            improved_bound = None
            for new_spec in new_spec_list:
                status, _ = self.verify_spec(new_spec, False)
                # print("splitted input ub ",t, new_spec.input_spec.input_ub)
                if status != Status.VERIFIED:
                    raise ValueError("The specifiction should be verified")
                if new_spec.get_perturbation_bound() is None:
                    raise ValueError("The perturbation bound should not be None")
                if improved_bound is None:
                    improved_bound = new_spec.get_perturbation_bound()
                    print("Improved bound", improved_bound)
                else:
                    improved_bound = min(improved_bound, new_spec.get_perturbation_bound())
                self.verified_spec_queue.put((new_spec.get_perturbation_bound(), new_spec))
            if improved_bound - curr_bound < tolerence_lim:
                return

                
            

    # def update_perturbation_bounds(self, verified_specs):
    #     for spec in verified_specs:
    #         # print("perturbation bound : ", spec.get_perturbation_bound())
    #         self.perturbation_bounds.append(spec.get_perturbation_bound())

    
    def compute_eta_norm(self, layer_index, debug_info=False):
        lb, ub = self.transformer.get_layer_bound(layer_index)
        if debug_info:
            print("Eta norm lower bound", lb)
            print("Eta norm upper bound", ub)
        temp_list = [max(abs(x), abs(y)) for x, y in zip(lb, ub)]
        eta_norm = torch.norm(torch.FloatTensor(temp_list))
        return eta_norm


    def run_parallel(self):
        """
        Parallel version of run
        TODO: refactor to remove duplicate code
        """
        split_score = self.set_split_score(self.init_prop, self.cur_specs)

        domain = util.get_domain(self.transformer)
        adv_label = self.adv_label

        while self.continue_search():
            self.update_depth()

            self.prev_lb = self.cur_lb
            self.reset_cur_lb()

            inputs = []

            # Gather inputs
            for spec in self.cur_specs:
                inputs.append((spec.input_spec, domain, adv_label))

            with Pool(processes=8) as pool:
                pool_return = pool.starmap(self.verify_node_parallel, inputs)

            # Process output
            for (status, _, lb), spec in zip(pool_return, self.cur_specs):
                self.update_cur_lb(lb)
                spec.status = status

                if status == Status.ADV_EXAMPLE:
                    self.global_status = status
                    return

                if self.is_timeout():
                    self.store_final_tree()
                    return

            print('Current Lower bound:', self.cur_lb)

            # Each spec should hold the prev lb and current lb
            self.cur_specs, verified_specs, chosen_split = self.cur_specs.prune(self.split, split_score=split_score,
                                                                                inp_template=self.inp_template,
                                                                                args=self.args, net=self.net)

            self.store_final_tree()

            # Update the tree size
            self.tree_size += len(self.cur_specs)

        self.check_verified_status()

    def store_final_tree(self):
        self.proof_tree = ProofTree(self.root_spec)
        self.template_store.add_tree(self.init_prop, self.proof_tree)


    def verify_node(self, transformer, prop, adv_label=None):
        """
        It is called from bnb_relu_complete. Attempts to verify (ilb, iub), there are three possible outcomes that
        are indicated by the status: 1) verified 2) adversarial example is found 3) Unknown
        """
        # print("Adv label", adv_label)
        lb, is_feasible, adv_ex = transformer.compute_lb(adv_label=adv_label, complete=True)
        status = self.get_status(adv_ex, prop, is_feasible, lb, self.net)
        unstable_relus = transformer.unstable_relus

        return status, unstable_relus, lb

    def verify_node_parallel(self, prop, domain, adv_label=None):

        transformer_builder = util.get_domain_builder(domain)
        transformer = transformer_builder(prop, complete=True)
        transformer = parse.get_transformer(transformer, self.net, prop)

        lb, is_feasible, adv_ex = transformer.compute_lb(adv_label=adv_label, complete=True)

        status = self.get_status(adv_ex, prop, is_feasible, lb, self.net)

        return status, transformer.unstable_relus, lb

    def get_status(self, adv_ex, prop, is_feasible, lb, net):
        status = Status.UNKNOWN
        if adv_ex is not None:
            if util.check_adversarial(adv_ex, net, prop):
                print("Found a counter example!")
                status = Status.ADV_EXAMPLE
            else:
                print("Found a spurious counter example!")
        elif (not is_feasible) or (lb is not None and torch.all(lb >= 0)):
            status = Status.VERIFIED
        return status

    def update_transformer(self, prop, relu_spec=None):
        relu_mask = None
        if relu_spec is not None:
            relu_mask = relu_spec.relu_mask

        if 'update_spec' in dir(self.transformer):
            self.transformer.update_spec(prop, relu_mask=relu_mask)
        else:
            transformer_builder = util.get_domain_builder(util.get_domain(self.transformer))
            self.transformer = transformer_builder(prop, complete=True)
            self.transformer = parse.get_transformer(self.transformer, self.net, prop)

    def check_verified_status(self):
        # Verified
        if len(self.cur_specs) == 0:
            self.global_status = Status.VERIFIED

    def reset_cur_lb(self):
        self.cur_lb = 1e7

    def is_timeout(self):
        return self.timeout is not None and (time.time() - self.init_time) > self.timeout

    def continue_search(self):
        return self.global_status == Status.UNKNOWN and len(self.cur_specs) > 0

    def update_cur_lb(self, lb):
        # lb can be None if the LP is infeasible
        if lb is not None:
            self.cur_lb = min(lb, self.cur_lb)

    def update_depth(self):
        print('Depth :', self.depth, 'Specs size :', len(self.cur_specs))
        self.depth += 1

    def create_initial_specs(self, prop, unstable_relus):
        if is_relu_split(self.split):
            relu_spec = specs.create_relu_spec(unstable_relus)
            self.root_spec = specs.Spec(prop, relu_spec=relu_spec, status=self.global_status)
            cur_specs = specs.SpecList([self.root_spec])
            config.write_log("Unstable relus: " + str(unstable_relus))
        else:
            if self.args.initial_split:
                # Do a smarter initial split similar to ERAN
                # This works only for ACAS-XU
                zono_transformer = ZonoTransformer(prop, complete=True)
                zono_transformer = parse.get_transformer(zono_transformer, self.net, prop)

                center = zono_transformer.centers[-1]
                cof = zono_transformer.cofs[-1]
                cof_abs = torch.sum(torch.abs(cof), dim=0)
                lb = center - cof_abs
                adv_index = torch.argmin(lb)
                input_len = len(prop.input_lb)
                smears = torch.abs(cof[:input_len, adv_index])
                split_multiple = 10 / torch.sum(smears)  # Dividing the initial splits in the proportion of above score
                num_splits = [int(torch.ceil(smear * split_multiple)) for smear in smears]

                inp_specs = prop.multiple_splits(num_splits)
                cur_specs = specs.SpecList([specs.Spec(prop, status=self.global_status) for prop in inp_specs])
                # TODO: Add a root spec in this case as well
            else:
                self.root_spec = specs.Spec(prop, status=self.global_status)
                cur_specs = specs.SpecList([self.root_spec])

        return cur_specs

    def set_split_score(self, prop, relu_mask_list, inp_template=None):
        """
        Computes relu score for each relu if the split method needs it. Otherwise, returns None
        """
        split_score = None
        if self.split == Split.RELU_GRAD:
            # These scores only work for torch models
            split_score = specs.score_relu_grad(relu_mask_list[0], prop, net=self.net)
        elif self.split == Split.RELU_ESIP_SCORE:
            # These scores only work with deepz transformer
            zono_transformer = ZonoTransformer(prop, complete=True)
            zono_transformer = parse.get_transformer(zono_transformer, self.net, prop)
            split_score = specs.score_relu_esip(zono_transformer)

        # Update the scores based on previous scores
        if inp_template is not None and split_score is not None:
            if type(self.args.pt_method) == PRUNE or type(self.args.pt_method) == REORDERING:
                # compute mean worst case improvements
                observed_split_scores = self.template_store.get_proof_tree(prop).get_observed_split_score()
                alpha = self.args.pt_method.alpha
                thr = self.args.pt_method.threshold

                for chosen_split in observed_split_scores:
                    if observed_split_scores[chosen_split] < self.args.pt_method.threshold:
                        split_score[chosen_split] = alpha*split_score[chosen_split] + (1-alpha)*(observed_split_scores[chosen_split] - thr)

        return split_score


# Write tests doing this instead
# def oval_bench(net_name):
#     pdprops = 'base_easy.pkl'
#     # pdprops = 'base_med.pkl'
#     # pdprops = 'base_hard.pkl'
#
#     path = 'tools/GNN_branching/cifar_exp/'
#     gt_results = pd.read_pickle(path + pdprops)
#     bnb_ids = gt_results.index
#
#     net = util.get_net(net_name, 'cifar10')
#
#     # batch ids were used for parallel processing in the original paper.
#     batch_ids = gt_results.index[0:10]
#
#     for new_idx, idx in enumerate(batch_ids):
#         imag_idx = gt_results.loc[idx]['Idx']
#         adv_label = gt_results.loc[idx]['prop']
#         eps_temp = gt_results.loc[idx]['Eps']
#
#         ilb, iub, true_label = util.ger_property_from_id(imag_idx, eps_temp, Dataset.CIFAR10)
#
#         status = bnb_complete(net, ilb, iub, true_label, LPTransformer, 'relu_grad', adv_label=adv_label)
#         print(status)
