import torch
import gurobipy as grb
import numpy as np

class RavenLPtransformer:
    def __init__(self, eps, inputs, batch_size, roll_indices, lb_coef, lb_bias, non_verified_indices,
                 lb_coef_dict, lb_bias_dict,
                 lb_penultimate_coef, lb_penultimate_bias, 
                 ub_penultimate_coef, ub_penultimate_bias,
                 lb_penult, ub_penult, constraint_matrices,
                 input_lbs, input_ubs,
                 disable_unrolling=False):
        def reshape(t):
            t = t.view(t.shape[0], t.shape[1], -1)
            return t
        self.device = 'cpu'
        if type(eps) is torch.Tensor:
            eps = torch.max(eps).item() 
        self.eps = eps

        self.inputs = inputs.to(self.device)
        # for topk single input is supported
        assert batch_size == 1
        self.batch_size = batch_size
        if roll_indices is not None:
            self.roll_indices = roll_indices.to(self.device).detach().numpy()
        else:
            self.roll_indices = None
        if non_verified_indices is not None:
            self.non_verified_indices = non_verified_indices.to(self.device).detach().numpy()
        else:
            self.non_verified_indices = None
        if lb_coef is not None:
            self.lb_coef = reshape(lb_coef.to(self.device))
        else:
            self.lb_coef = None
        if lb_bias is not None:
            self.lb_bias = lb_bias.to(self.device)
        else:
            self.lb_bias = None
        if lb_penultimate_coef is not None:
            self.lb_penultimate_coef = reshape(lb_penultimate_coef.to(self.device))
            self.lb_penultimate_bias = lb_penultimate_bias.to(self.device)
            self.ub_penultimate_coef = reshape(ub_penultimate_coef.to(self.device))
            self.ub_penultimate_bias = ub_penultimate_bias.to(self.device)
            self.lb_penult = lb_penult.to(self.device)
            self.ub_penult = ub_penult.to(self.device)
        else:
            self.lb_penultimate_coef = None
            self.lb_penultimate_bias = None
            self.ub_penultimate_coef = None
            self.ub_penultimate_bias = None
            self.lb_penult = None
            self.ub_penult = None

        self.constraint_matrices = constraint_matrices.to(self.device)
        self.input_lbs = input_lbs.view(input_lbs.shape[0], -1).detach().to(self.device)
        self.input_ubs = input_ubs.view(input_ubs.shape[0], -1).detach().to(self.device)
        self.disable_unrolling = disable_unrolling
        if lb_bias_dict is not None:
            self.lb_bias_dict = lb_bias_dict
            self.lb_coef_dict = lb_coef_dict
            self.process_dict()
        else:
            self.lb_bias_dict = None
            self.lb_coef_dict = None


        # Gurobi model
        self.gmdl = grb.Model()
        self.gmdl.setParam('OutputFlag', False)
        self.gmdl.setParam('TimeLimit', 300)
        self.gmdl.Params.MIPFocus = 3
        self.gmdl.Params.ConcurrentMIP = 3

        self.input_vars = None
        self.output_vars = None
        self.penult_vars = None
        self.penult_vars_activation = None
        self.final_ans = None
    
    def process_dict(self):
        for x in self.lb_coef_dict.values():
            for coef in x:
                coef = coef.detach().to(self.device)
                coef = coef.view(coef.shape[0], coef.shape[1], -1)
                coef = coef.numpy()

        for x in self.lb_bias_dict.values():
            for bias in x:
                bias = bias.detach().to(self.device)
                bias = bias.numpy()

    def input_constraints(self):
        assert self.inputs.shape[0] == self.batch_size
        self.inputs = self.inputs.view(self.batch_size, -1)
        if len(self.inputs) <= 0:
            return
        delta = self.gmdl.addMVar(self.inputs[0].shape[0], lb = -self.eps, ub = self.eps, vtype=grb.GRB.CONTINUOUS, name='uap_delta')
        self.input_vars = [self.gmdl.addMVar(self.inputs[i].shape[0], 
                                lb = self.input_lbs[i].detach().numpy(),
                                ub = self.input_ubs[i].detach().numpy(),
                                vtype=grb.GRB.CONTINUOUS, name=f'input_{i}')
                                for i in range(self.batch_size)]
        # ensure all inputs are perturbed by the same uap delta.
        for i, v in enumerate(self.input_vars):
            self.gmdl.addConstr(v == self.inputs[i].detach().numpy() + delta)   
    
    def output_len(self):
        if self.lb_coef is not None and len(self.lb_coef) > 0:
            return self.lb_coef[0].shape[0]
        else:
            for x in self.lb_coef_dict.values():
                for coef in x:
                    return coef.shape[0]
        raise ValueError(f'Can not find length of outputs')

    def output_variables(self):
        output_length = self.output_len()
        self.output_vars = [self.gmdl.addMVar(output_length, 
                        lb=-float('inf'), ub=float('inf'),
                        vtype=grb.GRB.CONTINUOUS, name=f'output_{i}')
                        for i in range(self.batch_size)]

    def penultimate_varibles(self):
        # assert self.lb_penultimate_coef.shape[0] == self.batch_size            
        self.penult_vars = [self.gmdl.addMVar(self.lb_penultimate_coef[idx].shape[0], 
                        lb=self.lb_penult[idx].detach().numpy(), ub=self.ub_penult[idx].detach().numpy(),
                        vtype=grb.GRB.CONTINUOUS, name=f'penult_{i}')
                        for i, idx in enumerate(self.roll_indices)]
        
        # np.maximum(self.x_lbs[i][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0]))
        # np.maximum(self.x_lbs[i][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0]))
        
        # Currnetly Hardcoded for Relu.
        self.penult_vars_activation = [self.gmdl.addMVar(self.lb_penultimate_coef[idx].shape[0], 
                        lb=np.maximum(self.lb_penult[idx].detach().numpy(), np.zeros(self.lb_penult[idx].shape[0])), 
                        ub=np.maximum(self.ub_penult[idx].detach().numpy(), np.zeros(self.ub_penult[idx].shape[0])),
                        vtype=grb.GRB.CONTINUOUS, name=f'penult_activation{i}')
                        for i, idx in enumerate(self.roll_indices)]



    def relu_constraints(self, pre_var, var, lb, ub):
        self.gmdl.addConstr(var >= pre_var)
        self.gmdl.addConstr(var >= 0)
        for i in range(lb.shape[0]):
            if lb[i] >= 0:
                self.gmdl.addConstr(var[i] <= pre_var[i])
            elif ub[i] <= 0:
                self.gmdl.addConstr(var[i] <= 0)
            else:
                lamb = ub[i]/(ub[i] - lb[i] + 1e-17)
                mu = - ((ub[i] * lb[i])/(ub[i] - lb[i] + 1e-17))
                self.gmdl.addConstr(var[i] <= lamb * pre_var[i] + mu)


    def penultimate_constraints(self, final_weight, final_bias):
        self.lb_penultimate_coef = self.lb_penultimate_coef.detach().numpy()
        self.lb_penultimate_bias = self.lb_penultimate_bias.detach().numpy()        
        self.ub_penultimate_coef = self.ub_penultimate_coef.detach().numpy()
        self.ub_penultimate_bias = self.ub_penultimate_bias.detach().numpy()

        print(f'roll indices {self.roll_indices}')
        for i, idx in enumerate(self.roll_indices):
            self.gmdl.addConstr(self.penult_vars[i] >= self.lb_penultimate_coef[idx] @ self.input_vars[idx] + self.lb_penultimate_bias[idx])
            self.gmdl.addConstr(self.penult_vars[i] <= self.ub_penultimate_coef[idx] @ self.input_vars[idx] + self.ub_penultimate_bias[idx])
            self.relu_constraints(pre_var=self.penult_vars[i], var=self.penult_vars_activation[i],
                                  lb=self.lb_penult[idx], ub=self.ub_penult[idx])
            constraint = self.constraint_matrices[i] @ final_weight
            bias =  self.constraint_matrices[i] @ final_bias
            constraint = constraint.detach().numpy()
            bias = bias.detach().numpy()
            self.gmdl.addConstr(self.output_vars[idx] >= constraint @ self.penult_vars_activation[i] + bias)


    def formulate_constriants(self, final_weight, final_bias):
        self.input_constraints()
        self.output_variables()
        if self.lb_penultimate_coef is not None:
            self.penultimate_varibles()
        final_weight = final_weight.to(self.device)
        final_bias = final_bias.to(self.device)
        self.lb_coef = self.lb_coef.detach().numpy()
        self.lb_bias = self.lb_bias.detach().numpy()
        for i in range(self.batch_size):
            if self.non_verified_indices is not None and i not in self.non_verified_indices:
                continue
            self.gmdl.addConstr(self.output_vars[i] >= self.lb_coef[i] @ self.input_vars[i] + self.lb_bias[i])

        if self.disable_unrolling is False:
            self.penultimate_constraints(final_weight=final_weight, final_bias=final_bias)
        return self

    def formulate_constriants_from_dict(self, final_weight, final_bias):
        self.input_constraints()
        self.output_variables()
        for i in range(self.batch_size):
            if self.non_verified_indices is not None and i not in self.non_verified_indices:
                continue
            if i not in self.lb_bias_dict.keys():
                continue
            for j, bias in enumerate(self.lb_bias_dict[i]):
                self.gmdl.addConstr(self.output_vars[i] >= self.lb_coef_dict[i][j].detach().cpu().numpy() @ self.input_vars[i] + bias.detach().cpu().numpy())
        return self



    def handle_optimization_res(self):
        if self.gmdl.status in [2, 6, 10]:
            # print("Final MIP gap value: %f" % self.gmdl.MIPGap)
            # try:
            #     print("Final MIP best value: %f" % self.final_ans.X)
            # except:
            #     print("No solution obtained")
            # print("Final ObjBound: %f" % self.gmdl.ObjBound)
            return self.gmdl.ObjBound
        else:
            if self.gmdl.status == 4:
                return 0.0
            elif self.gmdl.status in [9, 11, 13]:
                print("Suboptimal solution")

                print("Final MIP gap value: %f" % self.gmdl.MIPGap)
                try:
                    print("Final MIP best value: %f" % self.final_ans.X)
                except:
                    print("No solution obtained")
                print("Final ObjBound: %f" % self.gmdl.ObjBound)
                if self.gmdl.SolCount > 0:
                    return self.gmdl.ObjBound
                else:
                    return 0.0    
            print("Gurobi model status", self.gmdl.status)
            print("The optimization failed\n")            
            if self.gmdl.status == 3:
                # self.gmdl.computeIIS()
                # self.gmdl.write("./debug_logs/model.ilp") 
                pass

            return 0.0

    def containment(self, cross_ex_verified_tuple):
        assert self.non_verified_indices is not None
        for x in cross_ex_verified_tuple:
            if x not in self.non_verified_indices:
                return False
        return True

    def add_crossex_verified(self, cross_ex_verified_tuples, new_bs):
        for cross_ex_tuple in cross_ex_verified_tuples:
            if self.containment(cross_ex_tuple):
                self.gmdl.addConstr(grb.quicksum(new_bs[j] for j in cross_ex_tuple) >= 1.0)

    def solv_MILP(self, cross_ex_verified_tuples=None):
        bs = []
        assert self.batch_size == 1
        output_var_list = self.output_vars[0].tolist()
        for logit_diff in output_var_list:
            bs.append(self.gmdl.addVar(vtype=grb.GRB.BINARY, name=f'b{i}'))
            self.gmdl.addGenConstrIndicator(bs[-1], True, logit_diff >= -1e-10)
            self.gmdl.addGenConstrIndicator(bs[-1], False, logit_diff <= -1e-10)
        
        self.final_ans = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f'p')
        self.gmdl.addConstr(self.final_ans >= grb.quicksum(bs[i] for i in range(bs)))
        self.gmdl.update()
        self.gmdl.setObjective(self.final_ans, grb.GRB.MINIMIZE)
        self.gmdl.optimize()

        return self.handle_optimization_res()

    def solv_LP(self):
        self.final_ans = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name=f'lp_bound')
        for i, final_var in enumerate(self.output_vars):
            final_var_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_min_{i}')
            self.gmdl.addGenConstrMin(final_var_min, final_var.tolist())
            self.gmdl.addConstr(final_var_min <= self.final_ans)
        self.gmdl.update()
        self.gmdl.setObjective(self.final_ans, grb.GRB.MINIMIZE)
        self.gmdl.optimize()

        return self.final_ans.X