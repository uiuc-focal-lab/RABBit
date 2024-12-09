import torch
from queue import PriorityQueue
import src.util as util
import src.specs.spec as specs
import src.parse as parse
import time
import src.bnb.bnb as bnb

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from raven.src import config
from src.common import Status
from src.lirpaAnalysis import LirpaTransformer
from src.specs.out_spec import create_out_constr_matrix, OutSpecType
from src.common.result import Result, Results
from src.util import compute_2dlist_intersection
from src.proof_transfer.template import TemplateStore


class Analyzer:
    def __init__(self, args, net=None, template_store=None, pruning_args=None):
        """
        @param args: configuration arguments for the analyzer such as the network, domain, dataset, attack, count, dataset,
            epsilon and split
        """
        self.args = args
        self.net = net
        self.template_store = template_store
        self.timeout = args.timeout
        self.device = config.DEVICE
        self.transformer = None
        self.init_time = None
        self.prop_count = None
        self.pruning_args = pruning_args
        self.final_perturbation_bound = {}
        self.inactive_relu_list = None
        self.feature_lb = None
        self.feature_ub = None
        self.out_constraint_mat = None
        self.all_sparsification_res = {}

        if self.net is None:
            self.net = util.get_net(self.args.net, self.args.dataset)
        if self.template_store is None:
            self.template_store = TemplateStore()

    def get_constraint_mat(self):
        if self.args.prop_index is None and self.args.prop_count > 1:
            raise NotImplementedError("Not implemented")
        return self.out_constraint_mat

    def get_all_sparsification_result(self):
        return self.all_sparsification_res
        

    def update_inactive_relu_list(self, inative_relu_list):
        self.inactive_relu_list = compute_2dlist_intersection(self.inactive_relu_list, inative_relu_list)

    
    def update_final_perturbation_bound(self, layer_index, peturbation_bound, perturbation_scaling):
        if layer_index not in self.final_perturbation_bound.keys():
            self.final_perturbation_bound[layer_index] = PriorityQueue()
            self.final_perturbation_bound[layer_index].put((peturbation_bound / perturbation_scaling))
        else:
            self.final_perturbation_bound[layer_index].put((peturbation_bound / perturbation_scaling))

    def get_final_perturbation_bound(self, layer_index):
        if layer_index in self.final_perturbation_bound.keys():
            if self.pruning_args.accuracy_drop is not None:
                assert self.pruning_args.accuracy_drop < 1.0
                cutoff = self.prop_count * (1.0 - self.pruning_args.accuracy_drop)
                while self.final_perturbation_bound[layer_index].qsize() > max(cutoff, 2):
                    _ = self.final_perturbation_bound[layer_index].get()
            print("Qsize", self.final_perturbation_bound[layer_index].qsize())
            top = self.final_perturbation_bound[layer_index].get()
            self.final_perturbation_bound[layer_index].put(top)
            return top
        else:
            return None

    def get_inactive_relu_list(self):
        return self.inactive_relu_list

    def get_analyzed_net(self):
        return self.net

    def get_feature_bounds(self):
        return self.feature_lb, self.feature_ub

    def update_feature_bounds(self, f_lb, f_ub):
        if self.feature_lb is None:
            self.feature_lb = f_lb
        else:
            self.feature_lb = torch.minimum(self.feature_lb, f_lb)
        if self.feature_ub is None:
            self.feature_ub = f_ub
        else:
            self.feature_ub = torch.maximum(self.feature_ub, f_ub)


    def analyze(self, prop, index):
        if self.args.enable_sparsification:
            self.net = util.get_net(self.args.net, self.args.dataset)
        self.update_transformer(prop)
        tree_size = 1
        self.out_constraint_mat = prop.output_constr_mat()

        # Check if classified correctly
        if util.check_adversarial(prop.input, self.net, prop):
            return Status.MISS_CLASSIFIED, tree_size
        # Check Adv Example with an Attack
        if self.args.attack is not None:
            adv = self.args.attack.search_adversarial(self.net, prop, self.args)
            if util.check_adversarial(adv, self.net, prop):
                return Status.ADV_EXAMPLE, tree_size

        if self.args.split is None:
            status = self.analyze_no_split()
        elif self.args.split is None:
            status = self.analyze_no_split_adv_ex(prop)
        else:
            with torch.no_grad():
                bnb_analyzer = bnb.BnB(self.net, self.transformer, prop, self.args.split, self.template_store,
                                   timeout=self.timeout, args=self.args, pruning_args=self.pruning_args)
            if self.args.parallel:
                bnb_analyzer.run_parallel()
            else:
                bnb_analyzer.run()
            status = bnb_analyzer.global_status
            print("global status", bnb_analyzer.global_status)
            if status == Status.VERIFIED:
                # if sparsification enabled copy the sparsified features
                sparsification_res = bnb_analyzer.get_sparsification_result()
                self.all_sparsification_res[index] = sparsification_res

                perturbation_bound = bnb_analyzer.get_final_perturbation_bound()
                perturbation_scale = bnb_analyzer.get_scaling_factor_for_perturbation()
                print("Perturbation scale", perturbation_scale)
                if perturbation_bound is not None and perturbation_scale is not None:
                    # self.final_perturbation_bound[self.pruning_args.layers_to_prune[0]] = perturbation_bound
                    self.update_final_perturbation_bound(self.pruning_args.layers_to_prune[0], 
                                        peturbation_bound=perturbation_bound, perturbation_scaling=perturbation_scale)
                inactive_relu_list = bnb_analyzer.get_inactive_relu_list()
                if inactive_relu_list is not None:
                    self.update_inactive_relu_list(inactive_relu_list)
                f_lb, f_ub = bnb_analyzer.get_feature_layer_bound()
                if f_lb is not None and f_ub is not None:
                    self.update_feature_bounds(f_lb=f_lb, f_ub=f_ub)
                else:
                    print("Feature bound is None")

            tree_size = bnb_analyzer.tree_size
            print("problem status -- ", status)
        return status, tree_size

    def update_transformer(self, prop):
        if self.transformer is not None and 'update_input' in dir(self.transformer) \
                and prop.out_constr.constr_type == OutSpecType.LOCAL_ROBUST:
            self.transformer.update_input(prop)
        else:
            domain_builder = util.get_domain_builder(self.args.domain)
            self.transformer = domain_builder(prop, complete=True)
            self.transformer = parse.get_transformer(self.transformer, self.net, prop)

    def analyze_no_split_adv_ex(self, prop):
        # TODO: handle feasibility
        lb, _, adv_ex1 = self.transformer.compute_lb()
        adv_ex = None
        if util.check_adversarial(adv_ex1, self.net, prop):
            adv_ex = adv_ex1
        status = Status.UNKNOWN
        if torch.all(lb >= 0):
            status = Status.VERIFIED
        elif adv_ex is not None:
            status = Status.ADV_EXAMPLE
        print(lb)
        return status

    def analyze_no_split(self):
        lb = self.transformer.compute_lb()
        ub = self.transformer.compute_ub()
        status = Status.UNKNOWN
        if torch.all(lb >= 0):
            status = Status.VERIFIED
        print(lb, ub)
        return status

    def run_analyzer(self):
        """
        Prints the output of verification - count of verified, unverified and the cases for which the adversarial example
            was found
        """
        print('Using %s abstract domain' % self.args.domain)

        # Overwrite number of properties to load if prop index
        # is set.
        if self.args.prop_index is not None:
            self.args.count = self.args.prop_index + 1 

        props, inputs = specs.get_specs(self.args.dataset, spec_type=self.args.spec_type, count=self.args.count,
                                        eps=self.args.eps)
        self.prop_count = len(props)

        if util.is_lirpa_domain(self.args.domain):
           results = self.analyze_lirpa(props=props)
        else:
            results = self.analyze_domain(props)

        # results.compute_stats()
        # print('Results: ', results.output_count)
        # print('Average time:', results.avg_time)
        return results

    # There are multiple clauses in the inout specification
    # Property should hold on all the input clauses
    @staticmethod
    def extract_status(cl_status):
        for status in cl_status:
            if status != Status.VERIFIED:
                return status
        return Status.VERIFIED

    def analyze_domain(self, props):
        results = Results(self.args)
        for i in range(len(props)):
            if self.args.prop_index is not None:
                if i != self.args.prop_index:
                    continue
            if i in self.args.ignore_properties:
                print("***Property %d ***" % (i+1))
                continue
            print("************************** Proof %d *****************************" % (i+1))
            num_clauses = props[i].get_input_clause_count()
            clause_ver_status = []
            ver_start_time = time.time()

            for j in range(num_clauses):
                cl_status, tree_size = self.analyze(props[i].get_input_clause(j), i)
                clause_ver_status.append(cl_status)

            status = self.extract_status(clause_ver_status)
            print(status)
            ver_time = time.time() - ver_start_time
            results.add_result(Result(ver_time, status, tree_size=tree_size))

        return results

    # def analyze_lirpa(self, inputs):
    #     results = Results(self.args)
    #     ver_init_time = time.time()
    #     N = self.args.count
    #     mean, std = specs.get_mean_std(self.args.dataset)
    #     inputs = (inputs - mean) / std
    #     lirpa_model = BoundedModule(self.net, torch.empty_like(inputs), device=inputs.device)
    #     ptb = PerturbationLpNorm(norm=float("inf"), eps=self.args.eps)
    #     image = BoundedTensor(inputs, ptb)

    #     # TODO: fix this when the test are added. domain is not a string now
    #     C = create_out_constr_matrix(N, labels)

    #     olb, _ = lirpa_model.compute_bounds(x=(image,), method=domain.split()[0], C=C)
    #     count_verified = torch.all(olb >= 0, dim=1).sum().item()
    #     print(torch.all(olb >= 0, dim=1))

    #     total_time = time.time() - ver_init_time
    #     count = olb.shape[0]
    #     for i in range(count):
    #         results.add_result(Result(total_time / count, olb >= 0))
    #     return results



    def analyze_lirpa(self, props):
        results = Results(self.args)        
        lirpa_analyzer = LirpaTransformer(self.args.domain, self.args.dataset)
        count = 0 
        for prop in props:
            ver_start_time = time.time()
            print("Lirpa analysis property", count)
            verification_result = lirpa_analyzer.verify_property(prop, self.args)
            if verification_result == Status.VERIFIED:
                sparsification_result = lirpa_analyzer.get_sparsification_result()
                if sparsification_result is not None:
                    self.all_sparsification_res[count] = sparsification_result
            ver_time = time.time() - ver_start_time
            results.add_result(Result(ver_time, verification_result, tree_size=1))
            count += 1
        return results
