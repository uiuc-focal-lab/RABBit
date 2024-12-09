import torch
import src.util as util 
from src.uap_results import *
from src.common import Status
from src.domains.domain_transformer import domain_transformer
import time

class BaselineAnalyzerBackend:
    def __init__(self, props, net, args):
        self.props = props
        self.net = net 
        self.args = args

    # Return a list of baseline results containing all important
    # Metadata.
    def run(self):
        baseline_results = []
        for prop in self.props:
            start_time = time.time()
            assert prop.get_input_clause_count() == 1
            transformer = domain_transformer(net=self.net, prop=prop.get_input_clause(0), domain=self.args.individual_prop_domain)
            baseline_res = transformer.populate_baseline_verifier_result()
            time_diff = time.time() - start_time
            baseline_res.time = time_diff
            baseline_results.append(baseline_res)
        return baseline_results