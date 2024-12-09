import src.util as util
import src.parse as parse
from src.common import Domain

def domain_transformer(net, prop, domain):
    domain_builder = util.get_domain_builder(domain)
    transformer = domain_builder(prop, complete=False)
    transformer = parse.get_transformer(transformer, net, prop)
    return transformer
