# Stores result of verifiers designed for verifying linf
# property.
class BaselineVerifierRes:
    def __init__(self, input, layer_lbs, layer_ubs, final_lb, time=None, zono_center=None, 
                 zono_coef = None, lb_coef=None, lb_bias=None) -> None:
        self.input = input
        self.layer_lbs = layer_lbs
        self.layer_ubs = layer_ubs
        self.final_lb = final_lb
        self.time = time
        # Populated if underlying verifier is zonotope
        self.zono_center = zono_center
        self.zono_coef = zono_coef
        # Populated if underlying verifier is auto_lirpa crown/deeppoly etc.
        self.lb_coef = lb_coef
        self.lb_bias = lb_bias
