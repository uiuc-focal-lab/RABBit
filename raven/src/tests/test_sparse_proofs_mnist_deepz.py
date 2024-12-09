import src.proof_transfer.proof_transfer as pt
import src.proof_transfer.approximate as ap

from unittest import TestCase
from raven.src import config
from src.specs.property import InputSpecType
from src.bnb import Split
from src.common import Domain
from src.proof_transfer.pt_types import ProofTransferMethod, PRUNE, REORDERING
from src.common.dataset import Dataset

class TestMnistSparsification_01(TestCase):
    def test_deepz_standard(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.01,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60,output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_pgd_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'
        args = pt.TransferArgs(net=config.MNIST_CONV_PGD_03, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.01,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_diffai_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'        
        args = pt.TransferArgs(net=config.MNIST_FFN_DIFFAI, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.01,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


    def test_deepz_colt_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown-temp-test/'
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


    def test_deepz_crown_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'        
        args = pt.TransferArgs(net=config.MNIST_CROWN_IBP, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.01,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)




class TestMnistSparsification_02(TestCase):
    def test_deepz_standard(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-deepz/'
        args = pt.TransferArgs(net=config.MNIST_STANDARD_MODIFIED, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=300, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_pgd_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-deepz/'        
        args = pt.TransferArgs(net=config.MNIST_CONV_PGD_03, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_diffai_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-deepz/'
        args = pt.TransferArgs(net=config.MNIST_FFN_DIFFAI, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=30, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


    def test_deepz_colt_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-deepz/'
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=300, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


    def test_deepz_crown_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-deepz/'
        args = pt.TransferArgs(net=config.MNIST_CROWN_IBP_MODIFIED, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=300, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

class TestMnistSparsification_1(TestCase):
    def test_deepz_standard(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-deepz/'
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=40, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60,output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_pgd_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'
        args = pt.TransferArgs(net=config.MNIST_CONV_PGD_03, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=40, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_diffai_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'        
        args = pt.TransferArgs(net=config.MNIST_FFN_DIFFAI, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


    def test_deepz_colt_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'        
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LIRPA_CROWN, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


    def test_deepz_crown_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'        
        args = pt.TransferArgs(net=config.MNIST_CROWN_IBP, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


class TestMnistSparsification_2(TestCase):
    def test_deepz_pgd_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'
        args = pt.TransferArgs(net=config.MNIST_CONV_PGD_03, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.2,
                               split=Split.RELU_ESIP_SCORE, count=30, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_deepz_diffai_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'
        args = pt.TransferArgs(net=config.MNIST_FFN_DIFFAI, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.2,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


    def test_deepz_colt_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'
        args = pt.TransferArgs(net=config.MNIST_FFN_03, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.2,
                               split=Split.RELU_ESIP_SCORE, count=30, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)


    def test_deepz_crown_trained(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        output_dir='final_results/mnist-alpha-crown/'
        args = pt.TransferArgs(net=config.MNIST_CROWN_IBP, domain=Domain.LIRPA_CROWN_OPT, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.2,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, 
                               timeout=60, output_dir=output_dir, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)
