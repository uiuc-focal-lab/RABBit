from unittest import TestCase
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode
import os
import dill

class TestUAP(TestCase):
    def test_cifar_uap_diffai(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names= ['cifar10convSmallRELUDiffAI.onnx'],
                            count_per_prop=50, prop_count=10, eps=5.0/255,
                            threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                            baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                            refine_intermediate_bounds =True, optimize_layers_count=2, 
                            bounds_for_individual_refinement=True, dataloading_seed = 0,
                            parallelize_executions=False, greedy_tuple= True,
                            lp_threshold=-0.5, prop_idx= prop_id,
                            max_linear_apprx=6, full_alpha= True,
                            device='cuda', prune_before_refine = 10,
                            always_correct_cross_execution = False,
                            result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                            execution_count_dct = {2: 10 ,3: 10, 4:8, 5:8},  bias = 0.1, use_ib_refinement = True,  bab_timeout = 60, max_targetted= 4, 
                            num_targetted= 10, refinement_batch_size= 600,
                            store_time_trace= True, branching_method= 'babsr')    
                ver.adptiveRaven(raven_args=args)   

    def test_cifar_uap_standard(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=['convSmallRELU__Point.onnx'],
                    count_per_prop=50, prop_count=10, eps=1.0/255,
                    threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                    refine_intermediate_bounds =True, optimize_layers_count=2, 
                    bounds_for_individual_refinement=True, dataloading_seed = 0,
                    parallelize_executions=False,
                    lp_threshold=-0.5, greedy_tuple= True,
                    max_linear_apprx=6, bab_timeout = 60,
                    device='cuda', prop_idx = prop_id,
                    full_alpha= True, addl_prune = 15, branching_execution_counts= {2, 3, 4},
                    always_correct_cross_execution = False,
                    result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                    execution_count_dct = {2: 12, 3: 12, 4:8}, bias = 0.1, use_ib_refinement = True,  max_targetted= 4,
                    refinement_batch_size= 300, store_time_trace= True, branching_method= 'babsr')
                ver.adptiveRaven(raven_args=args)
        

    def test_cifar_uap_big_diffai(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=['cifar_convBigRELU__DiffAI.onnx'],
                    count_per_prop=50, prop_count=10, eps=2.0/255,
                    threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                    refine_intermediate_bounds =True, optimize_layers_count=2, 
                    bounds_for_individual_refinement=True, dataloading_seed = 0,
                    parallelize_executions=False,  greedy_tuple= True,
                    lp_threshold=-0.5, num_targetted= 10, 
                    max_linear_apprx=6, bab_timeout = 60,
                    device='cuda', prop_idx= prop_id,
                    full_alpha= False, prune_before_refine = 10,
                    always_correct_cross_execution = False,
                    result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                    execution_count_dct = {2: 10, 3: 8}, bias = 0.1, use_ib_refinement = True,  max_targetted= 3,
                    overall_batch_size= 6, bab_batch_size = 4, 
                    refinement_batch_size= 6, store_time_trace= True, branching_method= 'babsr')
                ver.adptiveRaven(raven_args=args)
            
        
            
                
               
    
    def test_cifar_uap_citrus(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=['citrus_convsmall_2.onnx'],
                            count_per_prop=50, prop_count=10, eps=2.0/255,
                            threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                            baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                            refine_intermediate_bounds =True, optimize_layers_count=2, 
                            bounds_for_individual_refinement=True, dataloading_seed = 0,
                            parallelize_executions=False, greedy_tuple= True,
                            lp_threshold=-0.5, prop_idx= prop_id,
                            max_linear_apprx=6, full_alpha= True,
                            device='cuda', prune_before_refine = 10,
                            always_correct_cross_execution = False,
                            result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                            execution_count_dct = {2: 14 ,3: 14, 4:10, 5:8},  bias = 0.1, use_ib_refinement = True,  bab_timeout = 60, max_targetted= 4, 
                            num_targetted= 10, refinement_batch_size= 600,
                            store_time_trace= True, branching_method= 'babsr')    
                ver.adptiveRaven(raven_args=args)


    def test_cifar_uap_sabr(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=['sabr_convsmall_2.onnx'],
                            count_per_prop=50, prop_count=10, eps=2.0/255,
                            threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                            baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                            refine_intermediate_bounds =True, optimize_layers_count=2, 
                            bounds_for_individual_refinement=True, dataloading_seed = 0,
                            parallelize_executions=False, greedy_tuple= True,
                            lp_threshold=-0.5, prop_idx= prop_id,
                            max_linear_apprx=6, full_alpha= True,
                            device='cuda', prune_before_refine = 10,
                            always_correct_cross_execution = False,
                            result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                            execution_count_dct = {2: 10 ,3: 10, 4:8, 5:8},  bias = 0.1, use_ib_refinement = True,  bab_timeout = 60, max_targetted= 4, 
                            num_targetted= 10, refinement_batch_size= 600,
                            store_time_trace= True, branching_method= 'babsr')    
                ver.adptiveRaven(raven_args=args)   

    def test_mnist_uap_diffai(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=['mnistconvSmallRELUDiffAI.onnx'],
                            count_per_prop=50, prop_count=10, eps=0.13, targetted_t_scale= 1.0, multiple_t_scale= 1.0, num_targetted= 20,
                            threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                            baseline_iteration=20, addl_prune= 0, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                            refine_intermediate_bounds =True, optimize_layers_count=2, 
                            bounds_for_individual_refinement=True, dataloading_seed = 0, greedy_tuple= True,
                            parallelize_executions=False, 
                            lp_threshold=-0.5,
                            max_linear_apprx=6, prop_idx= prop_id,
                            device='cuda', branching_execution_counts= {2, 3, 4},
                            always_correct_cross_execution = False,
                            result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                            execution_count_dct = {2: 24, 3: 20, 4: 18, 5: 8},  bias = 0.1, use_ib_refinement = True,  bab_timeout = 60, max_targetted= 5,
                            refinement_batch_size= 300, 
                            store_time_trace= True, branching_method= 'babsr')    
                ver.adptiveRaven(raven_args=args)


    
    def test_mnist_uap_sabr(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=['sabr_convsmall_1.onnx'],
                            count_per_prop=50, prop_count=10, eps=0.15, targetted_t_scale= 1.0, multiple_t_scale= 1.0, num_targetted= 24,
                            threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                            baseline_iteration=20, addl_prune= 0, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                            refine_intermediate_bounds =True, optimize_layers_count=2, 
                            bounds_for_individual_refinement=True, dataloading_seed = 0, greedy_tuple= True,
                            parallelize_executions=False, 
                            lp_threshold=-0.5,
                            max_linear_apprx=6, prop_idx= prop_id,
                            device='cuda', branching_execution_counts= {2, 3, 4},
                            always_correct_cross_execution = False,
                            result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                            execution_count_dct = {2: 24, 3: 20, 4: 18, 5: 8},  bias = 0.1, use_ib_refinement = True,  bab_timeout = 60, max_targetted= 5,
                            refinement_batch_size= 300, 
                            store_time_trace= True, branching_method= 'babsr')    
                ver.adptiveRaven(raven_args=args)

    
    def test_mnist_uap_standard(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=['mnist_convSmallRELU__Point.onnx'],
                            count_per_prop=50, prop_count=10, eps=0.07, targetted_t_scale= 1.0, multiple_t_scale= 1.0, num_targetted= 24,
                            threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                            baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                            refine_intermediate_bounds =True, optimize_layers_count=2, 
                            bounds_for_individual_refinement=True, dataloading_seed = 0, greedy_tuple= True,
                            parallelize_executions=False, 
                            lp_threshold=-0.5, addl_prune= 10,
                            max_linear_apprx=6, prop_idx= prop_id,
                            device='cuda', branching_execution_counts= {2, 3, 4},
                            always_correct_cross_execution = False,
                            result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                            execution_count_dct = {2: 24, 3: 20, 4: 18, 5: 8},  bias = 0.1, use_ib_refinement = True,  bab_timeout = 60, max_targetted= 5,
                            refinement_batch_size= 300, 
                            store_time_trace= True, branching_method= 'babsr')    
                ver.adptiveRaven(raven_args=args)


    def test_mnist_uap_citrus(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=['citrus_convsmall_1.onnx'],
                            count_per_prop=50, prop_count=10, eps=0.15, targetted_t_scale= 1.0, multiple_t_scale= 1.0, num_targetted= 24, addl_prune= 10,
                            threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                            baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                            refine_intermediate_bounds =True, optimize_layers_count=2, 
                            bounds_for_individual_refinement=True, dataloading_seed = 0, greedy_tuple= True,
                            parallelize_executions=False, 
                            lp_threshold=-0.5,
                            max_linear_apprx=6, prop_idx= prop_id,
                            device='cuda', branching_execution_counts= {2, 3, 4},
                            always_correct_cross_execution = False,
                            result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                            execution_count_dct = {2: 24, 3: 20, 4: 18, 5: 8},  bias = 0.1, use_ib_refinement = True,  bab_timeout = 60, max_targetted= 5,
                            refinement_batch_size= 300, 
                            store_time_trace= True, branching_method= 'babsr')    
                ver.adptiveRaven(raven_args=args)

