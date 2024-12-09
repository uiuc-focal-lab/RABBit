import torch
from src.uap_results import UAPSingleRes
from src.domains.diff_deeppoly import DiffDeepPoly
from src.uap_lp_new import UAPLPtransformer
import time
from src.common import Status

class UapDiff:
    def __init__(self, net, props, args, baseline_results) -> None:
        self.net = net
        self.props = props
        self.args = args
        self.total_props = len(self.props)
        self.baseline_results = baseline_results
        self.difference_lbs_dict = {}
        self.difference_ubs_dict = {}
        self.input_list = []
        self.eps = args.eps
        self.input_lbs = []
        self.input_ubs = []
        self.constr_matrices = []
        self.no_lp_for_verified = False
        self.baseline_verified_props = 0

    def compute_difference_dict(self):
        for i in range(len(self.baseline_results)):
            for j in range(i+1, len(self.baseline_results)):
                result1 = self.baseline_results[i]
                result2 = self.baseline_results[j]
                input1 = result1.input
                input1_lbs = result1.layer_lbs
                input1_ubs = result1.layer_ubs
                input2 = result2.input
                input2_lbs = result2.layer_lbs
                input2_ubs = result2.layer_ubs
                with torch.no_grad():
                    diff_poly_ver = DiffDeepPoly(input1=input1, input2=input2, net=self.net, 
                                lb_input1=input1_lbs, ub_input1=input1_ubs,
                                lb_input2=input2_lbs, ub_input2=input2_ubs, device='cpu')
                    delta_lbs, delta_ubs = diff_poly_ver.run()
                self.difference_lbs_dict[(i, j)] = delta_lbs
                self.difference_ubs_dict[(i, j)] = delta_ubs
            self.input_list.append(self.baseline_results[i].input)        

    def populate_lbs_and_ubs(self):
        for i in range(len(self.baseline_results)):
            self.input_lbs.append(self.baseline_results[i].layer_lbs)
            self.input_ubs.append(self.baseline_results[i].layer_ubs)

    def populate_matrices(self):
        for prop in self.props:
            self.constr_matrices.append(prop.get_input_clause(0).output_constr_mat())
    
    def prune_verified_props(self):
        new_props = []
        new_baseline_results = []
        for i, prop in enumerate(self.props):
            if torch.min(self.baseline_results[i].final_lb) >= 0.0:
                self.baseline_verified_props += 1
                continue
            new_props.append(prop)
            new_baseline_results.append(self.baseline_results[i])
        self.props = new_props
        self.baseline_results = new_baseline_results

    def run(self, proportion=False) -> UAPSingleRes:
        start_time = time.time()
        if self.no_lp_for_verified == True:
            self.prune_verified_props()
        self.populate_lbs_and_ubs()        
        self.compute_difference_dict()
        self.populate_matrices()
        if self.args.debug_mode is True:
            print("input1 ", self.input_list[0])
            print("input2 ", self.input_list[1])
        # Call the lp formulation with the differential lp code.
        uap_lp_transformer = UAPLPtransformer(mdl=self.net, xs=self.input_list, 
                                              eps=self.eps, x_lbs=self.input_lbs,
                                              x_ubs=self.input_ubs, d_lbs=self.difference_lbs_dict,
                                              d_ubs=self.difference_ubs_dict, constraint_matrices=self.constr_matrices,
                                              debug_mode=self.args.debug_mode,
                                              track_differences=self.args.track_differences)
        # Formulate the Lp problem.
        uap_lp_transformer.create_lp()
        verified_percentages = None
        global_lb = None
        verified_status = Status.UNKNOWN
        if proportion == False:
            global_lb = uap_lp_transformer.optimize_lp()
            print("Diff global lb", global_lb)
            if global_lb >= 0.0:
                verified_status = Status.VERIFIED            
        else:
            verified_percentages = uap_lp_transformer.optimize_milp_percent()
            verified_props = verified_percentages * len(self.props)
            verified_percentages = (verified_props + self.baseline_verified_props) / self.total_props
            print("Diff Verified percentages", verified_percentages)
            if verified_percentages >= self.args.cutoff_percentage:
                verified_status = Status.VERIFIED

        time_taken = time.time() - start_time
        return UAPSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                    status=verified_status, global_lb=None, time_taken=time_taken, 
                    verified_proportion=verified_percentages)
