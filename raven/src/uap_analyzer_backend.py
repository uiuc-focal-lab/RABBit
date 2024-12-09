import torch
import src.util as util 
from src.uap_results import *
from src.common import Status, Domain
from src.domains.domain_transformer import domain_transformer
from src.baseline_uap_verifier import BaselineAnalyzerBackend
from src.uap_domains.uap_domain_transformer import get_uap_domain_transformer
import time


class UAPAnalyzerBackendWrapper:
    def __init__(self, props, args) -> None:
        self.props = props
        self.args = args
        with torch.no_grad():
            self.net = util.get_net(self.args.net, self.args.dataset, debug_mode=self.args.debug_mode)    

    def get_radius(self):
        pass

    def update_props(self, eps):
        pass

    def get_opt_radius(self, radius_l, radius_r, use_uap_verifier, tolerence_lim=1e-7):
        pass
    

    def verify(self, use_uap_verifier=False):
        baseline_verfier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
        baseline_result = baseline_verfier.run()
        return None
    
    def run(self) -> UAPResult:
        # Baseline results correspond to running each property individually.
        with torch.no_grad():
            baseline_verfier = BaselineAnalyzerBackend(props=self.props, net=self.net, args=self.args)
            individual_verification_results = baseline_verfier.run()
        baseline_res = self.run_uap_verification(domain=self.args.baseline_domain, 
                                                 individual_verification_results=individual_verification_results)
        uap_algorithm_res = self.run_uap_verification(domain=self.args.domain, 
                                                 individual_verification_results=individual_verification_results)
        return UAPResult(baseline_res=baseline_res, UAP_res=uap_algorithm_res)


    def run_uap_verification(self, domain, individual_verification_results):
        uap_verifier = get_uap_domain_transformer(domain=domain, net=self.net, props=self.props, 
                                                           args=self.args, 
                                                           baseline_results=individual_verification_results)
        if self.args.no_lp_for_verified == True and domain == Domain.UAP_DIFF:
            uap_verifier.no_lp_for_verified = True
        return uap_verifier.run(proportion=self.args.compute_proportion)


# class UAPAnalyzerBackend:
#     def __init__(self, props, net, args, baseline_results):
#         self.props = props
#         self.net = net 
#         self.args = args
#         self.baseline_results = baseline_results

#     def run(self) -> UAPSingleRes:
#         start_time = time.time()
#         problem_status = Status.UNKNOWN
#         if self.coefs is None or self.centers is None:
#             raise ValueError("Coefs and centers must be both not null") 
#         with torch.no_grad():
#             transformer = domain_transformer(net=self.net, prop=self.props[0].get_input_clause(0), 
#                                          domain=self.args.domain)
#             transformer.set_coefs_centers(coefs=self.coefs, centers=self.centers)
#             lb = transformer.compute_lb()
#         print("UAP verifier lower bound", lb)
#         if lb >= 0:
#             problem_status = Status.VERIFIED
#         time_taken = time.time() - start_time
#         result = UAPSingleRes(domain=self.args.domain, 
#                                 input_per_prop=self.args.count_per_prop, status=problem_status, 
#                                 global_lb=lb, time_taken=time_taken)
#         return result
