from unittest import TestCase
from src.common.dataset import Dataset
from src.common import Domain
from src.bnb import Split
from src.analyzer import Analyzer
import raven.src.config as config
import ssl

# This is a hack to avoid SSL related errors while getting CIFAR10 data
from src.specs.input_spec import InputSpecType

ssl._create_default_https_context = ssl._create_unverified_context


class TestMNISTBox(TestCase):
    # 85% accuracy here with eps=0? Why is it not same as deepz/deeppoly
    def test_mlp_box_torch1(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0)
        Analyzer(args).run_analyzer()

    # ~100% accuracy here with eps=0?
    def test_mlp_box_onnx1(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.BOX, dataset=Dataset.MNIST, eps=0.0)
        Analyzer(args).run_analyzer()

    # 0% accuracy here with eps=1
    def test_mlp_box_onnx2(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.BOX, dataset=Dataset.MNIST, eps=1.0)
        Analyzer(args).run_analyzer()

    # 0% accuracy with eps=1 sanity check
    def test_mlp_box_torch2(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.BOX, dataset=Dataset.MNIST, eps=1.0)
        Analyzer(args).run_analyzer()


class TestMNISTDeepz(TestCase):
    # 98% accuracy here with eps=0 for sanity check
    # TODO: Something wrong with torch nets getting 86% here
    def test_mlp_deepz_torch1(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.0)
        Analyzer(args).run_analyzer()

    # ~100% accuracy here with eps=0 for sanity check
    def test_mlp_deepz_onnx1(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.0)
        Analyzer(args).run_analyzer()

    # 0% accuracy here with eps=1 for sanity check
    def test_mlp_deepz_torch2(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=1.0)
        Analyzer(args).run_analyzer()

    # 0% accuracy here with eps=1 for sanity check
    def test_mlp_deepz_onnx2(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=1.0)
        Analyzer(args).run_analyzer()

    # ~25% accuracy here with eps=0.03 for sanity check matched with ERAN
    def test_mlp_deepz_onnx3(self):
        args = config.Args(net=config.MNIST_FFN_L4, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.03)
        Analyzer(args).run_analyzer()

    # Input split is ineffective on high-dimensional images.
    # Thus this test does not converge.
    def test_mlp_deepz_split_onnx2(self):
        args = config.Args(net=config.MNIST_FFN_L4, domain=Domain.DEEPZ, dataset=Dataset.MNIST,
                           split=Split.INPUT)
        Analyzer(args).run_analyzer()

    # works but quite slow since proving 729 specs for each input
    # Also prints Proof too many times
    # Results: { < Status.VERIFIED: 1 >: 8, < Status.UNKNOWN: 3 >: 2}
    # Average time: 67.43715884685516
    def test_mlp_deepz_patch_split_onnx(self):
        args = config.Args(net=config.MNIST_FFN_L4, domain=Domain.DEEPZ, dataset=Dataset.MNIST,
                           split=Split.INPUT, attack=InputSpecType.PATCH)
        Analyzer(args).run_analyzer()


class TestMnistLinf(TestCase):
    # 98% accuracy here with eps=0 for sanity check
    def test_mlp_deeppoly_torch1(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, eps=0.0)
        Analyzer(args).run_analyzer()

    # ~100% accuracy here with eps=0 for sanity check
    def test_mlp_deeppoly_onnx1(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, eps=0.0)
        Analyzer(args).run_analyzer()

    # 0% accuracy here with eps=1 for sanity check
    def test_mlp_deeppoly_torch2(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, eps=1.0)
        Analyzer(args).run_analyzer()

    # 0% accuracy here with eps=1 for sanity check
    def test_mlp_deeppoly_onnx2(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, eps=1.0)
        Analyzer(args).run_analyzer()

    # TODO:
    def test_mlp_deepz_esip_onnx(self):
        args = config.Args(net=config.MNIST_FFN_L4, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.03,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()

    # ~40% accuracy here with eps=0.03 for sanity check matched with ERAN
    def test_mlp_deeppoly_onnx3(self):
        args = config.Args(net=config.MNIST_FFN_L4, domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, eps=0.03)
        Analyzer(args).run_analyzer()

    # 0 out of 100 cases need splitting
    def test_mlp_lp_torch(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.LP, dataset=Dataset.MNIST, eps=0.1,
                           split=Split.RELU_GRAD)
        Analyzer(args).run_analyzer()

    # VNN-COMP network
    # verified: 44 / 100
    # adv_example: 37 / 100
    # This becomes worse with only deepz + lp. deeppoly current implementation is fast on FC but not that fast
    # with conv
    def test_mlp_lp_esip_onnx1(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.LP, dataset=Dataset.MNIST, eps=0.03,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()

    # verified: 85 / 100
    # adv_example: 12 / 100
    def test_mlp_lp_esip_onnx2(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.LP, dataset=Dataset.MNIST, eps=0.02,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()

    # verified: 4 / 100
    # adv_example: 84 / 100
    def test_mlp_lp_esip_onnx3(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.LP, dataset=Dataset.MNIST, eps=0.05,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()

    def test_mlp_lp_esip_onnx4(self):
        args = config.Args(net=config.MNIST_FFN_L4, domain=Domain.LP, dataset=Dataset.MNIST, eps=0.03,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()

    # verified:  92 / 100
    def test_convSmall_deepz_onnx(self):
        args = config.Args(net=config.CIFAR_CONV_SMALL, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.03)
        Analyzer(args).run_analyzer()

    # verified:  93 / 100
    def test_convSmall_deeppoly_onnx(self):
        args = config.Args(net=config.CIFAR_CONV_SMALL, domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, eps=0.03)
        Analyzer(args).run_analyzer()

    # works but quite slow to figure out if splitting is needed
    def test_convSmall_lp_esip_onnx(self):
        args = config.Args(net=config.CIFAR_CONV_SMALL, domain=Domain.LP, dataset=Dataset.MNIST, eps=0.03,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()

    # Taken from https://github.com/eth-sri/colt/tree/master/trained_models/onnx
    # verified:  43 / 100
    def test_conv01_deepz_onnx(self):
        args = config.Args(net=config.MNIST_FFN_01, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.1)
        Analyzer(args).run_analyzer()

    # Lil slow as well 60/100
    def test_conv01_deeppoly_onnx(self):
        args = config.Args(net=config.MNIST_FFN_01, domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, eps=0.1)
        Analyzer(args).run_analyzer()

    # Too slow again on lp
    # verified: 68 / 100
    # adv_example: 11 / 100
    # Time taken: 2130.8400869369507
    def test_conv01_lp_esip_onnx(self):
        args = config.Args(net=config.MNIST_FFN_01, domain=Domain.LP, dataset=Dataset.MNIST, eps=0.1,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()

    # timeout = 30
    # verified: 18 / 100
    # adv_example: 79 / 100
    # Time
    # taken: 955.2926478385925

    # timeout = 200
    # verified: 18 / 100
    # adv_example: 79 / 100
    # Time
    # taken: 1478.9428930282593

    # timeout = 1000
    # verified: 18 / 100
    # adv_example: 79 / 100
    # Time
    # taken: 3870.425934076309
    def test_conv03_lp_esip_onnx(self):
        args = config.Args(net=config.MNIST_FFN_03, domain=Domain.LP, dataset=Dataset.MNIST, eps=0.1,
                           split=Split.RELU_ESIP_SCORE)
        res = Analyzer(args).run_analyzer()


class TestCIFAR10Linf(TestCase):
    # 53% accuracy here with eps=0
    def test_mlp_box_onnx1(self):
        args = config.Args(net=config.CIFAR_CONV_SMALL, domain=Domain.BOX, dataset=Dataset.CIFAR10, eps=0.0)
        Analyzer(args).run_analyzer()

    # 53% accuracy here with eps=0 for sanity check
    def test_mlp_deepz_onnx1(self):
        args = config.Args(net=config.CIFAR_CONV_SMALL, domain=Domain.DEEPZ, dataset=Dataset.CIFAR10, eps=1e-5)
        Analyzer(args).run_analyzer()

    # Extremely slow because of the convolutional layer
    #  accuracy here with eps=0 for sanity check
    # TODO: Make this faster
    def test_mlp_deeppoly_onnx1(self):
        args = config.Args(net=config.CIFAR_CONV_2_255, domain=Domain.DEEPPOLY, dataset=Dataset.CIFAR10, eps=0.0)
        Analyzer(args).run_analyzer()

    def test_conv01_lp_esip_onnx(self):
        args = config.Args(net=config.CIFAR_CONV_2_255, domain=Domain.LP, dataset=Dataset.CIFAR10, eps=2/255,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()

    def test_conv02_lp_esip_onnx(self):
        args = config.Args(net=config.CIFAR_CONV_8_255, domain=Domain.LP, dataset=Dataset.CIFAR10, eps=8/255,
                           split=Split.RELU_ESIP_SCORE)
        Analyzer(args).run_analyzer()


class TestMnistPatch(TestCase):
    def test_mlp_box_torch1(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.BOX, dataset=Dataset.MNIST, attack=InputSpecType.PATCH)
        Analyzer(args).run_analyzer()

    def test_mlp_deepz_torch1(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.DEEPZ, dataset=Dataset.MNIST, attack=InputSpecType.PATCH)
        Analyzer(args).run_analyzer()

    def test_mlp_deepz_torch2_complete(self):
        # Here timeout is for each patch
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.DEEPZ, dataset=Dataset.MNIST, attack=InputSpecType.PATCH,
                           split=Split.INPUT_SB, timeout=20)
        Analyzer(args).run_analyzer()

    def test_mlp_deeppoly_torch1(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.DEEPPOLY, dataset=Dataset.MNIST, attack=InputSpecType.PATCH)
        Analyzer(args).run_analyzer()

    # Not many deep proof trees
    def test_mlp2_deepz_complete(self):
        # Here timeout is for each patch
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.DEEPZ, dataset=Dataset.MNIST, attack=InputSpecType.PATCH,
                           split=Split.INPUT_SB, timeout=50)
        Analyzer(args).run_analyzer()


class TestAcasXu(TestCase):
    def test_acas1_box_onnx1(self):
        args = config.Args(net=config.ACASXU(1, 1), domain=Domain.BOX, dataset=Dataset.ACAS)
        Analyzer(args).run_analyzer()

    def test_acas1_deepz_onnx1(self):
        args = config.Args(net=config.ACASXU(1, 1), domain=Domain.DEEPZ, dataset=Dataset.ACAS)
        Analyzer(args).run_analyzer()

    def test_acas1_deeppoly_onnx1(self):
        args = config.Args(net=config.ACASXU(1, 1), domain=Domain.DEEPPOLY, dataset=Dataset.ACAS, split=Split.INPUT_SB, timeout=30)
        Analyzer(args).run_analyzer()

    def test_acas1_lp_esip_onnx(self):
        args = config.Args(net=config.ACASXU(1, 1), domain=Domain.DEEPZ, dataset=Dataset.ACAS,
                           split=Split.INPUT_SB, timeout=50, count=4)
        Analyzer(args).run_analyzer()

    # 5/10 verified with timeout=20
    def test_acas1_deepz_esip_onnx(self):
        args = config.Args(net=config.ACASXU(1, 3), domain=Domain.DEEPPOLY,
                           dataset=Dataset.ACAS,
                           split=Split.INPUT_GRAD, count=10, timeout=50, parallel=False)
        Analyzer(args).run_analyzer()
