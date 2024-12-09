
from unittest import TestCase
from raven.src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.uap_analyzer as analyzer
import src.uap_analyzer as uap_ver


class TestBasicUap(TestCase):
    def test_mnist_uap(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.MNIST, sink_label=None,
            spec_type=InputSpecType.UAP, count=1, count_per_prop=5, eps=0.05, net=config.MNIST_FFN_DIFFAI,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25,
            uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=False, write_file=True,
            no_lp_for_verified = True, debug_mode=False, track_differences=True)
        uap_ver.UapVerification(uap_verfication_args)

    def test_cifar_uap(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            individual_prop_domain=Domain.DEEPZ,
            domain=Domain.UAP_DIFF, baseline_domain=Domain.UAP_BASIC_LP, dataset=Dataset.CIFAR10, sink_label=None,
            spec_type=InputSpecType.UAP, count=20, count_per_prop=5, eps=3/255, net=config.CIFAR_CONV_DIFFAI,                                                                                                              
            timeout=100, output_dir='results/', radius_l=0.002, radius_r=0.25, 
            uap_mode=analyzer.UAPMode.VERIFICATION, compute_proportion=True, write_file=True,
            no_lp_for_verified = True, debug_mode=False, track_differences=True)
        uap_ver.UapVerification(uap_verfication_args)