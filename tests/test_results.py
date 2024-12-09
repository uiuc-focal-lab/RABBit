from unittest import TestCase
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode
import dill
import numpy as np
from pathlib import Path

  
def _get_main_result(meta_dir, prop_count, res_name):
    results = []
    for i in range(prop_count):
      with open(f'{meta_dir}/{res_name}_prop={i}.dill', 'rb') as file:
        meta_loaded_object = dill.load(file)
        obj = meta_loaded_object.results_lst[0]
        results.append(max([tup[0] for t, tup in obj.verified_dict.items() if t < obj.raven_args.total_time]))

    print('RABBit mean results: ', np.mean(results))
      

def main_result(results_dir, net, eps, prop_count, count_per_prop, res_name = 'results'):
  meta_dir = get_exp_path(results_dir, net, eps, prop_count, count_per_prop, verify_type = 'meta')
  if Path(meta_dir).exists():
    _get_main_result(meta_dir, prop_count, res_name)
  else:
    print(f'skipping main results for {meta_dir} as not found')
 
def get_exp_path(result_dir, net, eps, prop_count, count_per_prop, verify_type = 'meta'):
  return f'{result_dir}/{net}/eps={eps}/prop_count={prop_count}/imgs={count_per_prop}/verify_{verify_type}/'

def main(results_dir = 'results/crossex_complete/', mode = 'main_result', net = 'cifar10convSmallRELUDiffAI.onnx',  eps = 5.0, prop_count = 10, count_per_prop = 50):
    eval(mode)(results_dir, net, eps, prop_count, count_per_prop)

class TestResults(TestCase):
    def test_analyze_main_results_cifar_sabr(self):
        print()
        main_result(results_dir = 'results/crossex_complete/', net = 'sabr_convsmall_2.onnx',  eps = 2.0, prop_count = 10, count_per_prop = 50)

    def test_analyze_main_results_mnist_sabr(self):
        print()
        main_result(results_dir = 'results/crossex_complete/', net = 'sabr_convsmall_1.onnx',  eps = 0.15, prop_count = 10, count_per_prop = 50)