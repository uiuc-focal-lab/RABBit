from unittest import TestCase
import sys
sys.path.append('RABBit/')
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode
import os


net_names = [config.CIFAR_CONV_SMALL]
eps = 1.0

args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
    count_per_prop=40, prop_count=1, eps=eps/255,
    threshold_execution=5, cross_executional_threshold=4, maximum_cross_execution_count=4, 
    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
    refine_intermediate_bounds =True, optimize_layers_count=2, 
    bounds_for_individual_refinement=True, dataloading_seed = 0,
    parallelize_executions=False,
    lp_threshold=-0.5,
    max_linear_apprx=4,
    device='cuda:1',
    always_correct_cross_execution = False,
    result_dir='icml_results_new', write_file=True, complete_verification= True, use_lp_bab= False, meta_scheduling= True)

if os.path.exists(args.file_path):
    os.remove(args.file_path)

ver.adptiveRaven(raven_args=args)