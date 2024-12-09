import src.proof_transfer.proof_transfer as pt
import src.proof_transfer.approximate as ap

from unittest import TestCase
from raven.src import config
from src.specs.property import InputSpecType
from src.bnb import Split
from src.common import Domain
from src.proof_transfer.pt_types import ProofTransferMethod, PRUNE, REORDERING
from src.common.dataset import Dataset


class TestReordering(TestCase):
    def test_conv01_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.01,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(0, 0.01),
                               timeout=100)
        pt.proof_transfer(args)

    def test_conv01_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(0, 0.01),
                               timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(0, 0.01),
                               timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=10, pt_method=REORDERING(0, 0.01),
                               timeout=10)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_prune30(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(30), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(0, 0.01),
                               timeout=200)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_random_1e3(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Random(1e-3), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=1, pt_method=REORDERING(0, 0.01),
                               timeout=200)
        pt.proof_transfer(args)


class TestReusing(TestCase):
    def test_conv01_lp_esip_onnx_int8(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=0, accuracy_drop=0.12)
        enable_sparsification = True
        store_in_file = True
        args = pt.TransferArgs(net=config.MNIST_STANDARD_MODIFIED, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=10, pt_method=ProofTransferMethod.REUSE, 
                               timeout=30, output_dir='results/', prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_conv01_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_prune30(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(30), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE,
                               timeout=200)
        pt.proof_transfer(args)


class TestCompleteAll(TestCase):
    def test_conv01_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL, timeout=200)
        pt.proof_transfer(args)

    def test_conv01_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL, timeout=200)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL, timeout=200)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL, timeout=200)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_prune30(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(30), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL,
                               timeout=200)
        pt.proof_transfer(args)


class TestCompletePruneCIFAR(TestCase):
    def test_conv_small_lp_esip_cifar_int16(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=False, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=2)
        enable_sparsification = True
        store_in_file = False         
        args = pt.TransferArgs(net=config.CIFAR_CROWN_IBP, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=10, pt_method=ProofTransferMethod.REUSE, 
                               timeout=200, prop_index=None, ignore_properties=[], 
                               pruning_args=pruning_args, enable_sparsification=enable_sparsification, store_in_file=store_in_file)
        pt.proof_transfer(args)

    def test_conv_big_lp_esip_cifar_int16(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_BIG, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=PRUNE(0, 0.005), timeout=400)
        pt.proof_transfer(args)

    def test_conv_small_lp_esip_cifar_int8(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=PRUNE(0, 0.005), timeout=400)
        pt.proof_transfer(args)

    def test_conv_big_lp_esip_cifar_int8(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_BIG, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=PRUNE(0, 0.005), timeout=400)
        pt.proof_transfer(args)


class TestCompletePatch(TestCase):
    def test_conv_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST,
                               attack=InputSpecType.PATCH, split=Split.INPUT, pt_method=ProofTransferMethod.ALL,
                               timeout=200)
        pt.proof_transfer(args)

    def test_conv_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST,
                               attack=InputSpecType.PATCH, split=Split.INPUT, pt_method=ProofTransferMethod.ALL,
                               timeout=30)
        pt.proof_transfer(args)


class TestCompleteAcas(TestCase):
    def test_deepz_acas_onnx_int8(self):
        pruning_args = config.PruningArgs(desried_perturbation=12.0, layers_to_prune=[-1], swap_layers=True, node_wise_bounds=False, 
                unstructured_pruning=True, structured_pruning=False, maximum_iteration=10)
        # desried_perturbation=None, layers_to_prune=None, swap_layers= False, node_wise_bounds=False,
        #                 unstructured_pruning=True, structured_pruning=False, maximum_iteration=10)
        args = pt.TransferArgs(domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.ACAS, split=Split.INPUT_SB, pt_method=ProofTransferMethod.REUSE, timeout=30, ignore_properties=[1], pruning_args=pruning_args)
        pt.proof_transfer_acas(args)

    def test_deepz_acas_onnx_int16(self):
        args = pt.TransferArgs(domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.ACAS, split=Split.INPUT_SB, pt_method=PRUNE(0, 0.01), timeout=20)
        pt.proof_transfer_acas(args)


class TestPrune(TestCase):
    def test_conv01_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=PRUNE(0, 0.003),
                               timeout=100)
        pt.proof_transfer(args)

    def test_conv01_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=PRUNE(0, 0.003),
                               timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=PRUNE(0, 0.003),
                               timeout=30)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=10, pt_method=PRUNE(0, 0.003),
                               timeout=10)
        pt.proof_transfer(args)

    def test_oval_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net="oval21/cifar_wide_kw.onnx", domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=8/255,
                               count=20, pt_method=PRUNE(0, 0.003), split=Split.RELU_ESIP_SCORE,
                               timeout=20)
        pt.proof_transfer(args)