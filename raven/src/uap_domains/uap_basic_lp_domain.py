import torch
from src.uap_results import UAPSingleRes
import gurobipy as grb
from src.common import Status

class UapBasicLP:
    def __init__(self, net, props, args, baseline_results) -> None:
        self.net = net
        self.props = props 
        self.args = args
        self.baseline_results = baseline_results
        self.batch_size = len(self.baseline_results)
        self.input_size = self.baseline_results[0].input.shape[0]
        self.baseline_lbs = []
        # zonotope baseline
        self.zono_centers = None
        self.zono_coefs = None
        # crown baseline
        self.lb_coefs = None
        self.lb_biases = None
        # Model
        self.model = grb.Model()
        # stores minimum of each individual property.
        self.prop_mins = []

    def populate_info(self):
        for res in self.baseline_results:
            self.baseline_lbs.append(torch.min(res.final_lb))
            if res.zono_center is not None:
                if self.zono_centers is None:
                    self.zono_centers = [res.zono_center]
                else:
                    self.zono_centers.append(res.zono_center)

            if res.zono_coef is not None:
                if self.zono_coefs is None:
                    self.zono_coefs = [res.zono_coef]
                else:
                    self.zono_coefs.append(res.zono_coef)

            if res.lb_coef is not None:
                if self.lb_coefs is None:
                    self.lb_coefs = [res.lb_coef]
                else:
                    self.lb_coefs.append(res.lb_coef)

            if res.lb_bias is not None:
                if self.lb_biases is None:
                    self.lb_biases = [res.lb_bias]
                else:
                    self.lb_biases.append(res.lb_bias)

    def formulate_zono_lb(self):
        if self.zono_coefs is None or self.zono_centers is None:
            raise ValueError("coefs or center is NULL.")
        assert len(self.zono_coefs) == len(self.zono_centers)
        self.model.setParam('OutputFlag', False)
        actual_coefs = []
        lbs = []
        for i, coef in enumerate(self.zono_coefs):
            center = self.zono_centers[i]
            input_coefs = coef[:self.input_size]
            input_coefs = input_coefs.T
            other_coefs = coef[self.input_size:]
            cof_abs = torch.sum(torch.abs(other_coefs), dim=0)
            lb = center - cof_abs
            lbs.append(lb.detach().numpy())
            actual_coefs.append(input_coefs)
        
        epsilons = self.model.addMVar(self.input_size, lb=-1, ub=1, name='epsilons')
        individual_lbs = []
        
        for i, input_coefs in enumerate(actual_coefs):
            input_coefs = input_coefs.detach().numpy()
            t = self.model.addMVar(input_coefs.shape[0], lb=float('-inf'), ub=float('inf'), name=f'individual_lbs_{i}')
            self.model.addConstr(input_coefs @ epsilons + lbs[i] == t)
            individual_lbs.append(t)
            var_min = self.model.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'var_min_{i}')
            self.model.addGenConstrMin(var_min, t.tolist())
            self.prop_mins.append(var_min)
            self.model.update()

    def run_zono_lp_baseline(self, proportion):
        self.baseline_lbs.sort()
        print("DeepZ lbs", self.baseline_lbs)
        self.formulate_zono_lb()
        if proportion == False:
            problem_min = self.model.addVar(lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                name='problem_min')
            self.model.addGenConstrMax(problem_min, self.prop_mins)
            self.model.setObjective(problem_min, grb.GRB.MINIMIZE)
            self.model.optimize()
            if self.model.status == 2:
                return problem_min.X
            else:
                print("Gurobi model status", self.model.status)
                self.model.computeIIS()
                self.model.write("model_baseline.ilp")
                return None
        else:
            binary_vars = []
            for i, var_min in enumerate(self.prop_mins):
                binary_vars.append(self.model.addVar(vtype=grb.GRB.BINARY, name=f'b{i}')) 
                # BIG M formulation 
                BIG_M = 1e11

                # Force binary_vars[-1] to be '1' when t_min > 0
                self.model.addConstr(BIG_M * binary_vars[-1] >= var_min)


                # Force binary_vars[-1] to be '0' when t_min < 0 or -t_min  > 0
                self.model.addConstr(BIG_M * (binary_vars[-1] - 1) <= var_min)
            p = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='p')
            self.model.addConstr(p == grb.quicksum(binary_vars[i] for i in range(self.batch_size)) / self.batch_size)
            # self.model.reset()
            self.model.update()
            self.model.setObjective(p, grb.GRB.MINIMIZE)
            self.model.optimize()
            if self.model.status == 2:
                return p.X
            else:
                print("Gurobi model status", self.model.status)
                self.model.computeIIS()
                self.model.write("model_basic.ilp")
                return None


    def run_crown_lp_baseline(self, proportion):
        pass                  

    def run(self, proportion=False) -> UAPSingleRes:
        self.populate_info()
        if self.zono_centers is not None:
            ans = self.run_zono_lp_baseline(proportion=proportion)
            if ans is None:
                return None
            else:
                global_lb = None
                verified_proportion = None
                verified_status = Status.UNKNOWN
                if proportion == False:
                    print("Baseline global lp ", ans)
                    global_lb = ans
                    if global_lb >= 0:
                        verified_status = Status.VERIFIED
                else:
                    print("Baseline proportion ", ans)
                    verified_proportion = ans
                    if verified_proportion >= self.args.cutoff_percentage:
                        verified_status = Status.VERIFIED

            return UAPSingleRes(domain=self.args.domain, input_per_prop=self.args.count_per_prop,
                    status=verified_status, global_lb=global_lb, time_taken=None, 
                    verified_proportion=verified_proportion)
        elif self.lb_coefs is not None:
            return self.run_crown_lp_baseline(proportion=proportion)