import torch
from src.common import Domain
from src.uap_domains.uap_basic_domain import UapBasic
from src.uap_domains.uap_basic_lp_domain import UapBasicLP
from src.uap_domains.uap_diff_domain import UapDiff

def get_uap_domain_transformer(domain, net, props, args, baseline_results):
    if domain is Domain.UAP_BASIC:
        transformer = UapBasic(net=net, props=props, args=args, baseline_results=baseline_results)
    elif domain is Domain.UAP_BASIC_LP:
        transformer = UapBasicLP(net=net, props=props, args=args, baseline_results=baseline_results)
    elif domain is Domain.UAP_DIFF:
        transformer = UapDiff(net=net, props=props, args=args, baseline_results=baseline_results)
    else:
         raise ValueError(f"Unrecognized UAP domain {domain}") 
    return transformer