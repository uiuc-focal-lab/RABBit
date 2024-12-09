import src.proof_transfer.proof_transfer as pt
import src.proof_transfer.approximate as ap

from unittest import TestCase
from raven.src import config
from src.bnb import Split
from src.common import Domain
from src.proof_transfer.param_tune import write_result
from src.proof_transfer.pt_types import ProofTransferMethod, PRUNE, REORDERING
from src.common.dataset import Dataset


class TestPLDI(TestCase):
    def test1(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0.003, 0.003),
                               timeout=100)
        pt.proof_transfer(args)

    def test2(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 0.003),
                               timeout=100)
        pt.proof_transfer(args)

    def test3(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 0.007),
                               timeout=100)
        pt.proof_transfer(args)

    def test4(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 0.007),
                               timeout=100)
        pt.proof_transfer(args)

    def test5(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 1e-4), timeout=100)
        pt.proof_transfer(args)

    def test6(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 1e-4), timeout=100)
        pt.proof_transfer(args)

    def test7(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 1e-4), timeout=400)
        pt.proof_transfer(args)

    def test8(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 1e-4), timeout=400)
        pt.proof_transfer(args)

    def test9(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 1e-4), timeout=400)
        pt.proof_transfer(args)

    def test10(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(0, 1e-4), timeout=400)
        pt.proof_transfer(args)


class TestReuse(TestCase):
    def test1(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE,
                               timeout=100)
        pt.proof_transfer(args)

    def test2(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE,
                               timeout=100)
        pt.proof_transfer(args)

    def test3(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE,
                               timeout=100)
        pt.proof_transfer(args)

    def test4(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE,
                               timeout=100)
        pt.proof_transfer(args)

    def test5(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test6(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test7(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, timeout=400)
        pt.proof_transfer(args)

    def test8(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, timeout=400)
        pt.proof_transfer(args)

    def test9(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, timeout=400)
        pt.proof_transfer(args)

    def test10(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.REUSE, timeout=400)
        pt.proof_transfer(args)


class TestReorder(TestCase):
    def test1(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0.003, 0.003),
                               timeout=100)
        pt.proof_transfer(args)

    def test2(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0.003, 0.003),
                               timeout=100)
        pt.proof_transfer(args)

    def test3(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0, 0.007),
                               timeout=100)
        pt.proof_transfer(args)

    def test4(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0, 0.007),
                               timeout=100)
        pt.proof_transfer(args)

    def test5(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0, 1e-4), timeout=100)
        pt.proof_transfer(args)

    def test6(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0, 1e-4), timeout=100)
        pt.proof_transfer(args)

    def test7(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0, 1e-4), timeout=400)
        pt.proof_transfer(args)

    def test8(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0, 1e-4), timeout=400)
        pt.proof_transfer(args)

    def test9(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0, 1e-4), timeout=400)
        pt.proof_transfer(args)

    def test10(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=REORDERING(0, 1e-4), timeout=400)
        pt.proof_transfer(args)


class TestSensitivity(TestCase):
    def test1(self):
        thrs = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        alphas = [0, 0.25, 0.5, 0.75, 1]

        for thr in thrs:
            for alpha in alphas:
                args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                                       dataset=Dataset.MNIST, eps=0.02,
                                       split=Split.RELU_ESIP_SCORE, count=100, pt_method=PRUNE(alpha, thr),
                                       timeout=100)
                sp = pt.proof_transfer(args)
                write_result(alpha, sp, thr)



