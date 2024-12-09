import torch
from raven.src import config
from copy import deepcopy
from src.analyzer import Analyzer
from src.common import Status
from src.util import get_linear_layers
from src.common.result import Result, Results
from src.proof_transfer.pt_util import compute_speedup, plot_verification_results
from src.proof_transfer.pt_types import ProofTransferMethod
from src.proof_transfer.prune_network import *
from src.proof_transfer.check_acas_xu_accuracy import check_acas_accuracy
from src.proof_transfer.approximate import check_accuracy



class TransferArgs:
    def __init__(self, domain, approx, pt_method=None, count=None, eps=0.01, dataset='mnist', attack='linf',
                 split=None, net='', timeout=30, output_dir='', prop_index = None, ignore_properties=[], pruning_args=None, 
                 enable_sparsification=False, store_in_file=True):
        self.net = config.NET_HOME + net
        self.net_name = net
        self.domain = domain
        self.pt_method = pt_method
        self.count = count
        self.eps = eps
        self.dataset = dataset
        self.attack = attack
        self.split = split
        if len(output_dir) == 0:
            self.output_dir = 'results/sparse_proof/'
        else:
            self.output_dir = output_dir
        self.output_dir= output_dir
        self.approximation = approx
        self.timeout = timeout
        self.prop_index = prop_index
        self.ignore_properties = ignore_properties
        self.pruning_args = pruning_args
        self.acas_xu_i = None
        self.acas_xu_j = None
        self.enable_sparsification = enable_sparsification
        self.store_in_file = store_in_file

    def set_acas_xu_indices(self, i, j):
        self.acas_xu_i = i
        self.acas_xu_j = j

    def get_acas_xu_indices(self):
        return self.acas_xu_i, self.acas_xu_j 

    def set_net(self, net):
        self.net = config.NET_HOME + net

    def get_verification_arg(self):
        arg = config.Args(net=self.net, domain=self.domain, dataset=self.dataset, eps=self.eps,
                          split=self.split, count=self.count, pt_method=self.pt_method, timeout=self.timeout)
        # net is set correctly again since the home dir is added here
        arg.net = self.net
        arg.net_name = self.net_name
        arg.ignore_properties = self.ignore_properties
        arg.prop_index = self.prop_index
        arg.enable_sparsification = self.enable_sparsification
        return arg
    
    def get_pruning_args(self):
        return self.pruning_args


def proof_transfer(pt_args):
    res, res_pt = proof_transfer_analyze(pt_args)
    speedup = 1.0
    # speedup = compute_speedup(res, res_pt, pt_args)
    # print("Proof Transfer Speedup :", speedup)
    # # plot_verification_results(res, res_pt, pt_args)
    return speedup


def proof_transfer_acas(pt_args):
    res = Results(pt_args)
    res_pt = Results(pt_args)
    for i in range(2, 3):
        for j in range(1, 9):
            pt_args.set_net(config.ACASXU(i, j))
            pt_args.set_acas_xu_indices(i, j)
            pt_args.count = 4
            r, rp = proof_transfer_analyze(pt_args)
            res.results_list += r.results_list
            res_pt.results_list += rp.results_list

    # compute merged stats
    res.compute_stats()
    res_pt.compute_stats()

    speedup = compute_speedup(res, res_pt, pt_args)
    print("Proof Transfer Speedup :", speedup)
    plot_verification_results(res, res_pt, pt_args)
    return speedup

def build_unsturctured_pruning_arguments(pt_args, analyzer, net=None):
    pruning_args = pt_args.get_pruning_args()
    if pruning_args is None:
        return None
    layer_index = pruning_args.layers_to_prune[0]
    perturbation_bound = analyzer.get_final_perturbation_bound(layer_index)
    print("Perturbation Bound", perturbation_bound)
    if net is None:
        net = analyzer.get_analyzed_net()
    network_pruning_args = PruneArgs(epsilon=perturbation_bound, net=net, layer_index=layer_index)
    return network_pruning_args

def build_sturctured_pruning_arguments(pt_args, analyzer, net=None):
    pruning_args = pt_args.get_pruning_args()
    if pruning_args is None:
        return None
    layer_index = pruning_args.layers_to_prune[0]
    if net is None:
        net = analyzer.get_analyzed_net()
    perturbation_bound = analyzer.get_final_perturbation_bound(layer_index)
    passive_relus = analyzer.get_inactive_relu_list()
    # print("No of passive relus last layer", len(passive_relus[-1]))
    # print("passive relus ", passive_relus[-1])
    network_pruning_args = PruneArgs(epsilon=perturbation_bound, net=net, layer_index=layer_index, 
                                unstructured_prune=False, passive=True, passive_relus=passive_relus)
    return network_pruning_args

def swap_pruning_layers(pruning_args):
    if len(pruning_args.layers_to_prune) == 0:
        return
    if pruning_args.layers_to_prune[0] == -1:
        pruning_args.layers_to_prune[0] = -2
    else:
        pruning_args.layers_to_prune[0] = -1


def store_pruning_stats(pt_args, stat_dictionary, accuracy_list):
    i, j = pt_args.get_acas_xu_indices()
    if i is None or j is None:
        return
    structured_pruning = pt_args.get_pruning_args().structured_pruning
    postfix = "structured" if structured_pruning else "unstructured_only"
    filename = 'results/pruning_stats/results_net_{}_{}-{}.txt'.format(i, j, postfix)
    filename_acc = 'results/pruning_stats/accuracy_net_{}_{}_{}.txt'.format(i, j, postfix)
    print("File Name ",filename)
    stat_file = open(filename, 'a+')
    acc_file = open(filename_acc, 'a+')
    for _, stat in stat_dictionary.items():
        stat_file.write('{}\n'.format(stat))
    acc_file.write('{}\n'.format(accuracy_list))
    stat_file.close()
    acc_file.close()

def compute_accuracy_stats(pt_args, pruned_net, accuracy_stat):
    i, j = pt_args.get_acas_xu_indices()
    if i is not None and j is not None: 
        accuracy = check_acas_accuracy(pt_args, pruned_net)
        print("Returned accuracy", accuracy)
        accuracy_stat.append(accuracy)
        if len(accuracy_stat) > 0:
            print("Accuracy", accuracy_stat[-1])


def network_pruning(pt_args, analyzer):
    stat_dictionary = {}
    template_store = analyzer.template_store
    accuracy_stats = []    
    for i in range(pt_args.get_pruning_args().maximum_iteration):
        prune_network_args = build_unsturctured_pruning_arguments(pt_args=pt_args, analyzer=analyzer)
        pruned_network = get_pruned_network(prune_args=prune_network_args)
        if pt_args.get_pruning_args().structured_pruning:
            structured_prune_network_args = build_sturctured_pruning_arguments(pt_args=pt_args, analyzer=analyzer, net=pruned_network)
            pruned_network = get_pruned_network(prune_args=structured_prune_network_args) 
        # Print and store stats of pruning.
        # compute_accuracy_stats(pt_args=pt_args, pruned_net=pruned_network, accuracy_stat=accuracy_stats)
        check_accuracy(analyzer.get_analyzed_net(), dataset=pt_args.dataset)
        print_stats(pruned_network, stat_dictionary=stat_dictionary)
        if pt_args.get_pruning_args().swap_layers and i < 6:
            swap_pruning_layers(pt_args.get_pruning_args())
        approx_args = pt_args.get_verification_arg()
        analyzer = Analyzer(approx_args, net=pruned_network, template_store=template_store, pruning_args=pt_args.get_pruning_args())
        res_pt = analyzer.run_analyzer()
        template_store = analyzer.template_store
    # Prune the network using the bounds obsereved in the last 
    # iteration.
    prune_network_args = build_unsturctured_pruning_arguments(pt_args=pt_args, analyzer=analyzer)
    pruned_network = get_pruned_network(prune_args=prune_network_args)
    structured_prune_network_args = build_sturctured_pruning_arguments(pt_args=pt_args, analyzer=analyzer, net=pruned_network)
    pruned_network = get_pruned_network(prune_args=structured_prune_network_args)
    check_accuracy(analyzer.get_analyzed_net(), dataset=pt_args.dataset)
    # compute_accuracy_stats(pt_args=pt_args, pruned_net=pruned_network, accuracy_stat=accuracy_stats)    
    print_stats(pruned_network, stat_dictionary=stat_dictionary)
    store_pruning_stats(pt_args, stat_dictionary, accuracy_stats)

def print_sparsification_result(file, property_no, result_dict, store_in_file):
    if store_in_file:
        file.write("property no {}\n".format(property_no))
        file.write("Initial sparsity {}\n".format(result_dict["Initial sparsity"]))
        file.write("Optimal Sparsity {}\n".format(result_dict["Optimal Sparsity"]))
        file.write("zero indices {}\n".format(result_dict["zero indices"]))
        file.write("Indices prune {}\n".format(result_dict["Indices prune"]))
        file.write("Remaining indices {}\n".format(result_dict["Remaining indices"]))
    else:
        print("property no {}\n".format(property_no))
        print("Initial sparsity {}\n".format(result_dict["Initial sparsity"]))
        print("Optimal Sparsity {}\n".format(result_dict["Optimal Sparsity"]))
        print("zero indices {}\n".format(result_dict["zero indices"]))
        print("Indices prune {}\n".format(result_dict["Indices prune"]))
        print("Remaining indices {}\n".format(result_dict["Remaining indices"]))



def compute_sparse_network(analyzer, zero_fetures_indices, non_zero_indices, pt_args, file):
    initial_sparsity = non_zero_indices.size()[0]
    l = 0
    r = initial_sparsity - 1
    opt_sparsity = 0
    print("Initial sparsity", initial_sparsity)
    org_net = deepcopy(analyzer.get_analyzed_net())
    last_layer = get_linear_layers(org_net)[-1]
    template_store = analyzer.template_store
    prune_last_layer(last_layer.weight, zero_fetures_indices)
    while l <= r:
        mid = (l + r) // 2
        print("Mid", mid)
        if mid <= 0:
            break
        print(mid)
        net = analyzer.get_analyzed_net()
        curr_last_layer = get_linear_layers(net)[-1]
        curr_last_layer.weight = deepcopy(last_layer.weight)
        indices_to_prune = non_zero_indices[:mid]
        prune_last_layer(curr_last_layer.weight, indices_to_prune)
        approx_args = pt_args.get_verification_arg()
        analyzer = Analyzer(approx_args, net=net, template_store=None, pruning_args=pt_args.get_pruning_args())
        try:
            res = analyzer.run_analyzer()
        except:
            break
        res_list = res.results_list
        if len(res_list) < 1:
            raise ValueError("Empty results returned")
        else:
            if res_list[0].ver_output == Status.VERIFIED:
                opt_sparsity = max(opt_sparsity, mid)
                print("Optimum sparsity", opt_sparsity)
                l = mid + 1
            else:
                r = mid - 1

    sparsification_result = {}
    sparsification_result["Initial sparsity"] = initial_sparsity
    sparsification_result["Optimal Sparsity"] = initial_sparsity - opt_sparsity
    sparsification_result["zero indices"] = zero_fetures_indices
    sparsification_result["Indices prune"] = non_zero_indices[:opt_sparsity]
    sparsification_result["Remaining indices"] = non_zero_indices[opt_sparsity:]
    print_sparsification_result(file, pt_args.prop_index, sparsification_result, pt_args.store_in_file)



def compute_position(analyzer, pt_args, file):
    last_layer = get_linear_layers(analyzer.get_analyzed_net())[-1]
    out_constraint_mat = analyzer.get_constraint_mat()
    out_constraint_mat = out_constraint_mat.T
    final_wt = out_constraint_mat @ last_layer.weight
    final_wt = torch.abs(final_wt)
    wt_bounds = torch.max(final_wt, dim=0)
    wt_bounds = wt_bounds[0]

    f_lb, f_ub = analyzer.get_feature_bounds()
    if f_lb is None or f_ub is None:
        return
    abs_feature = torch.maximum(torch.abs(f_lb), torch.abs(f_ub))
    greedy_features = torch.mul(abs_feature, wt_bounds)
    sorted_features = torch.sort(greedy_features)
    nonzero_count = torch.count_nonzero(sorted_features[0])
    zero_fetures = sorted_features[0][:-nonzero_count]
    zero_fetures_indices = sorted_features[1][:-nonzero_count]
    nonzero_fetures = sorted_features[0][-nonzero_count:]
    nonzero_fetures_indices = sorted_features[1][-nonzero_count:]
    compute_sparse_network(analyzer, zero_fetures_indices, nonzero_fetures_indices, pt_args, file)


def get_file_name_from_pt_args(pt_args):
    filename = pt_args.output_dir+"{}_{}_{}.dat".format(pt_args.net_name, pt_args.domain, pt_args.eps)
    print(filename)
    return filename



def proof_transfer_analyze(pt_args):
    args = pt_args.get_verification_arg()
    count = pt_args.count
    filename = get_file_name_from_pt_args(pt_args)
    file = open(filename, "a+")
    if pt_args.enable_sparsification:
        args.prop_index = None
        pt_args.prop_index = None
        analyzer = Analyzer(args, net=None, template_store=None, pruning_args=pt_args.get_pruning_args())
        res_pt = analyzer.run_analyzer()
        all_sparsification_res = analyzer.get_all_sparsification_result()
        for k in all_sparsification_res:
            print_sparsification_result(file, k, all_sparsification_res[k],  pt_args.store_in_file)
    else:
        for i in range(count):
            args.prop_index = i
            pt_args.prop_index = i
            analyzer = Analyzer(args, net=None, template_store=None, pruning_args=pt_args.get_pruning_args())
            res_pt = analyzer.run_analyzer()
            if len(res_pt.results_list) > 0 and res_pt.results_list[0].ver_output == Status.VERIFIED :
                compute_position(analyzer=analyzer, pt_args=pt_args, file=file)
    file.close()
    # print("Proof Transfer", analyzer.get_feature_bounds())
    # check_accuracy(analyzer.get_analyzed_net(), dataset=pt_args.dataset)
    template_store = analyzer.template_store
    res = res_pt
    return res, res_pt


def get_reordered_template_store(args, template_store):
    args.pt_method = ProofTransferMethod.REORDERING
    # Compute reordered template
    analyzer_reordering = Analyzer(args, template_store=template_store)
    # TODO: make this as a separate a function from run_analyzer that takes some budget for computing
    _ = analyzer_reordering.run_analyzer()
    # This template store should contain leaf nodes of reordered tree
    template_store = analyzer_reordering.template_store
    return template_store


def get_perturbed_network(pt_args):
    # Generate the approximate network
    approx_net = pt_args.approximation.approximate(pt_args.net, pt_args.dataset)
    return approx_net
