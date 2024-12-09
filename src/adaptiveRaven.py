import torch
from src.common import Dataset, RavenMode
from src.specLoader import get_specification, get_std
from src.netLoader import get_net
from src.adaptiveRavenBackend import AdaptiveRavenBackend
from src.adaptiveRavenResult import AdaptiveRavenResultList
from raven.src.config import mnist_data_transform
from src.metaScheduler import MetaScheduler
from src.metaResult import OverallResult
import os
import dill
from tqdm import tqdm
from onnx_converter import load_resnet

class RavenArgs:
    def __init__(self, raven_mode : RavenMode, dataset : Dataset, net_names,
                count_per_prop=None, prop_count=None, eps=None,
                threshold_execution=5, cross_executional_threshold=4, 
                maximum_cross_execution_count=3, gscale = 600, baseline_iteration=10,
                refinement_iterations=30, unroll_layers = False, addl_prune  = 0, unroll_layer_count=3,
                optimize_layers_count = None, full_alpha=False,
                bounds_for_individual_refinement=True,
                always_correct_cross_execution=False,
                parallelize_executions = False, lp_threshold=None,
                max_linear_apprx=3,
                populate_trace=False, prop_idx = None,
                device=None, prune_before_refine = 0,
                refine_intermediate_bounds = False, dataloading_seed = 0, greedy_tuple = False, gcp_crown = False,
                result_dir=None, write_file=True, complete_verification = False, use_lp_bab = False, verify_mode = 'meta', execution_count_dct = {2: 10, 3: 8, 4: 8}, 
                bias = 0.1, branching_execution_counts = {2, 3, 4, 5}, max_targetted = 4, max_branching_execution_count_dct = {2:1, 3:1, 4:1, 5:1}, overall_batch_size = 512, bab_batch_size = 64, bab_timeout = 120, refinement_batch_size = 300, timeout_seconds = 5, use_ib_refinement = True, num_targetted = None, store_time_trace = False, 
                targetted_t_scale = 1.0, multiple_t_scale = 1.0, branching_method = 'babsr', res_name = 'results') -> None:
        self.raven_mode = raven_mode
        self.dataset = dataset
        self.net_names = net_names
        assert len(self.net_names) > 0
        if raven_mode in [RavenMode.UAP, RavenMode.UAP_TARGETED]:
            assert len(self.net_names) == 1
        self.count_per_prop = count_per_prop
        self.prop_count = prop_count
        self.greedy_tuple = greedy_tuple
        self.eps = eps
        self.prune_before_refine = prune_before_refine
        self.threshold_execution = threshold_execution
        self.gcp_crown = gcp_crown
        self.cross_executional_threshold = cross_executional_threshold
        self.maximum_cross_execution_count = maximum_cross_execution_count
        self.baseline_iteration = baseline_iteration
        self.refinement_iterations = refinement_iterations
        self.bounds_for_individual_refinement=True 
        self.full_alpha = full_alpha
        self.prop_idx = prop_idx
        self.store_time_trace = store_time_trace
        self.unroll_layers = unroll_layers
        self.unroll_layer_count = unroll_layer_count
        self.always_correct_cross_execution = always_correct_cross_execution
        self.parallelize_executions = parallelize_executions
        self.refine_intermediate_bounds = refine_intermediate_bounds
        self.optimize_layers_count = optimize_layers_count
        self.lp_threshold = lp_threshold
        self.gscale = gscale
        self.res_name = res_name
        self.max_linear_apprx = max_linear_apprx
        self.populate_trace = populate_trace
        self.bias = bias
        self.targetted_t_scale = targetted_t_scale
        self.multiple_t_scale = multiple_t_scale
        self.timeout_seconds = timeout_seconds
        self.use_ib_refinement = use_ib_refinement
        self.overall_batch_size = overall_batch_size
        self.refinement_batch_size = min(overall_batch_size, refinement_batch_size)
        self.bab_batch_size = bab_batch_size
        self.total_time = count_per_prop * bab_timeout
        self.branching_execution_counts = branching_execution_counts
        self.max_branching_execution_count_dct = max_branching_execution_count_dct
        if populate_trace:
            self.always_correct_cross_execution = True
            print(f'always compute trace {self.always_correct_cross_execution}')
        self.dataloading_seed = dataloading_seed
        self.device = device
        self.bab_timeout = bab_timeout
        self.addl_prune = addl_prune
        self.result_dir = result_dir
        self.write_file = write_file
        self.complete_verification = complete_verification
        self.use_lp_bab = use_lp_bab
        self.skip_torch_init = False
        self.execution_count_dct = execution_count_dct
        self.max_targetted = max_targetted
        self.branching_method = branching_method
        self.num_targetted = num_targetted if num_targetted is not None else max(self.execution_count_dct.values())
        self.run_topk, self.run_abc, self.meta_scheduling = False, False, False 
        if verify_mode == 'topk':
            self.run_topk = True 
        elif verify_mode in ('meta', 'cbab', 'target'):
            self.meta_scheduling = True
        elif verify_mode == 'abc':
            self.run_abc = True 
        else:
            raise ValueError('invalid verify_mode')
        self.verify_mode = f'verify_{verify_mode}'
        if self.dataset == Dataset.CIFAR10:
            self.file_dir = f'results/crossex_complete/{self.net_names[0]}/eps={round(self.eps * 255, 2)}/prop_count={self.prop_count}/imgs={self.count_per_prop}/{self.verify_mode}/'
        else:
            self.file_dir = f'results/crossex_complete/{self.net_names[0]}/eps={self.eps}/prop_count={self.prop_count}/imgs={self.count_per_prop}/{self.verify_mode}/'
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

class Property:
    def __init__(self, inputs, labels, eps, constraint_matrices, lbs, ubs) -> None:
        self.inputs = inputs
        self.labels = labels
        self.eps = eps
        self.constraint_matrices = constraint_matrices
        self.lbs = lbs
        self.ubs = ubs

def adptiveRaven(raven_args : RavenArgs):
    if 'resnet' in raven_args.net_names[0]:
        raven_args.skip_torch_init = True
        nets = [load_resnet(net_name) for net_name in raven_args.net_names]
    else:
        nets = get_net(net_names = raven_args.net_names, dataset = raven_args.dataset)
    total_input_count = raven_args.prop_count * raven_args.count_per_prop
    images, labels, constraint_matrices, lbs, ubs = get_specification(dataset=raven_args.dataset,
                                                            raven_mode=raven_args.raven_mode, 
                                                            count=total_input_count, nets=nets, eps=raven_args.eps,
                                                            dataloading_seed=raven_args.dataloading_seed,
                                                            net_names=raven_args.net_names)

    assert len(raven_args.net_names) > 0
    assert images.shape[0] == raven_args.count_per_prop * raven_args.prop_count
    assert labels.shape[0] == raven_args.count_per_prop * raven_args.prop_count
    assert constraint_matrices.shape[0] == raven_args.count_per_prop * raven_args.prop_count

    result_list = AdaptiveRavenResultList(args=raven_args) if (not raven_args.complete_verification) else OverallResult()
    data_transform = mnist_data_transform(dataset=raven_args.dataset, net_name=raven_args.net_names[0])
    print(f'net name {raven_args.net_names[0]} data transform {data_transform}')
    prop_iter = range(raven_args.prop_count) if raven_args.prop_idx is None else [raven_args.prop_idx]
    for i in tqdm(prop_iter):
        start = i * raven_args.count_per_prop
        end = start + raven_args.count_per_prop
        prop_images, prop_labels, prop_constraint_matrices = images[start:end], labels[start:end], constraint_matrices[start:end]
        prop_lbs, prop_ubs = lbs[start:end], ubs[start:end]
        prop = Property(inputs=prop_images, labels=prop_labels, 
                        eps=raven_args.eps / get_std(dataset=raven_args.dataset, transform=data_transform),
                        constraint_matrices=prop_constraint_matrices, lbs=prop_lbs, ubs=prop_ubs)
        
        if raven_args.complete_verification: 
            verifier = MetaScheduler(prop=prop, nets=nets, args=raven_args)
            ver_mode = 'verify_meta' if raven_args.verify_mode in ('verify_cbab', 'verify_target') else raven_args.verify_mode
            getattr(verifier, ver_mode)()
            result_list.add_res(res = verifier.property_result)
            if (i == raven_args.prop_idx) and raven_args.complete_verification:
                with open(raven_args.file_dir + f'{raven_args.res_name}_prop={i}.dill', "wb") as file:
                    dill.dump(result_list, file)
            # try:
            #     srsly.write_json(raven_args.file_dir + f'{raven_args.res_name}_prop={i}.json', verifier.property_result.verified_dict)
            # except:
            #     pass
            if (i == raven_args.prop_idx):
                return
        else:
            verifier = AdaptiveRavenBackend(prop=prop, nets=nets, args=raven_args)
            result = verifier.verify(raven_args.complete_verification)
            result_list.add_res(res=result)
    
    if raven_args.branching_method == 'babsr':       
        file_name = raven_args.file_dir + f'{raven_args.res_name}.dill'
    else:
        file_name = raven_args.file_dir + f'{raven_args.res_name}_{raven_args.branching_method}.dill'
    if os.path.exists(file_name):
        os.remove(file_name)
    if raven_args.complete_verification:
        with open(file_name, "wb") as file:
            dill.dump(result_list, file)
    else:
        result_list.analyze()
        

    

