import gurobipy as grb
import torch
from torch import nn
from src.common.network import Network, LayerType
import numpy as np

class UAPLPtransformer:
    def __init__(self, mdl, xs, eps, x_lbs, x_ubs, d_lbs, d_ubs, 
                 constraint_matrices, debug_mode=False, track_differences=True):
        self.mdl = mdl
        self.xs = xs
        self.batch_size = len(xs)
        self.input_size = None
        self.shape = None
        if len(xs) > 0:
            self.input_size = xs[0].shape[0]
            self.set_shape()
        # maximum allowed perturbation.
        self.eps = eps
        self.x_lbs = x_lbs
        self.x_ubs = x_ubs
        self.d_lbs = d_lbs
        self.d_ubs = d_ubs
        # this tracks the index of the last linear (fully connected or conv) 
        # layer already. encountered its -1 at the beginning.
        self.linear_layer_idx = -1 
        self.gmdl = grb.Model()
        self.gurobi_variables = []
        self.debug_log_filename = './debug_logs/uap_lp_debug_log.txt'
        self.debug = True
        self.debug_log_file = None
        self.constraint_matrices = constraint_matrices
        self.tolerence = 2*1e-1
        self.debug_mode = debug_mode
        self.track_differences = track_differences
    
    def set_shape(self):
        if self.input_size == 784:
            self.shape = (1, 28, 28)
        elif self.input_size == 3072:
            self.shape = (3, 32, 32)
        else:
            raise ValueError("Unsupported dataset!")



    def create_lp(self):
        if self.batch_size <= 0:
            return
        self.debug_log_file = open(self.debug_log_filename, 'w+')
        self.gmdl.setParam('OutputFlag', False)
        self.gmdl.setParam('TimeLimit', 12*60)
        self.create_constraints()

    def optimize_lp(self):
        assert len(self.constraint_matrices) == self.batch_size
        if self.batch_size <= 0:
            return 0.0
        final_vars = []
        final_min_vars = []
        for i in range(self.batch_size):
            constraint_mat = self.constraint_matrices[i]
            final_var = self.gmdl.addMVar(constraint_mat.shape[1], lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                            name=f'final_var_{i}')
            self.gmdl.addConstr(final_var == constraint_mat.T.detach().numpy() @ self.gurobi_variables[-1]['vs'][i])
            final_vars.append(final_var)
            final_var_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_min_{i}')
            self.gmdl.addGenConstrMin(final_var_min, final_var.tolist())
            final_min_vars.append(final_var_min)
        problem_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                        name='problem_min')
        self.gmdl.addGenConstrMax(problem_min, final_min_vars)
        self.gmdl.setObjective(problem_min, grb.GRB.MINIMIZE)
        if self.debug_mode == True:
            self.gmdl.write("./debug_logs/model.lp")
            # self.gmdl.write("./debug_logs/out.sol")
        
        self.gmdl.optimize()
        if self.gmdl.status == 2:
            self.debug_log_file.write(f"Problem min {problem_min.X}\n")
            # print(f"Problem min {problem_min.X}\n")
            self.debug_log_file.close()
            return problem_min.X
        else:
            if self.gmdl.status == 4:
                self.gmdl.setParam('PreDual',0)
                self.gmdl.setParam('DualReductions', 0)
                self.gmdl.optimize()
            elif self.gmdl.status == 13 or self.gmdl.status == 9:
                print("Suboptimal solution")
                self.debug_log_file.close()
                if self.gmdl.SolCount > 0:
                    return problem_min.X
                else:
                    return 0.0
            print("Gurobi gmndl status", self.gmdl.status)
            self.debug_log_file.close()
            if self.gmdl.status == 3:
                self.gmdl.computeIIS()
                self.gmdl.write("model.ilp") 
                self.debug_log_file.close()   
                return NotImplementedError
        
    
    def optimize_milp_percent(self):
        assert len(self.constraint_matrices) == self.batch_size
        if self.batch_size <= 0:
            return 0.0
        bs = []
        final_vars = []
        final_min_vars = []
        for i, constraint_mat in enumerate(self.constraint_matrices):
            final_var = self.gmdl.addMVar(constraint_mat.shape[1], lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS, 
                                            name=f'final_var_{i}')
            self.gmdl.addConstr(final_var == constraint_mat.T.detach().numpy() @ self.gurobi_variables[-1]['vs'][i])
            final_vars.append(final_var)
            final_var_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_min_{i}')
            self.gmdl.addGenConstrMin(final_var_min, final_var.tolist())
            final_min_vars.append(final_var_min)
            bs.append(self.gmdl.addVar(vtype=grb.GRB.BINARY, name=f'b{i}'))
            # Binary encoding (Big M formulation )
            BIG_M = 1e11

            # Force bs[-1] to be '1' when t_min > 0
            self.gmdl.addConstr(BIG_M * bs[-1] >= final_var_min)

            # Force bs[-1] to be '0' when t_min < 0 or -t_min  > 0
            self.gmdl.addConstr(BIG_M * (bs[-1] - 1) <= final_var_min)
        
        p = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f'p')
        self.gmdl.addConstr(p == grb.quicksum(bs[i] for i in range(self.batch_size)) / self.batch_size)
        # self.gmdl.reset()
        self.gmdl.update()
        self.gmdl.setObjective(p, grb.GRB.MINIMIZE)
        self.gmdl.optimize()
        
        if self.debug_mode is True:
            print("Here")
            self.gmdl.write("./debug_logs/model.lp")
            # self.gmdl.write("./debug_logs/out.sol")
        
        if self.gmdl.status == 2:
            self.debug_log_file.write(f"proportion {p.X}\n")
            # print(f"verified proportion {p.X}\n")
            self.debug_log_file.close()
            return p.X
        else:
            if self.gmdl.status == 4:
                self.gmdl.setParam('PreDual',0)
                self.gmdl.setParam('DualReductions', 0)
                self.gmdl.optimize()
            elif self.gmdl.status == 13 or self.gmdl.status == 9:
                print("Suboptimal solution")
                self.debug_log_file.close()
                if self.gmdl.SolCount > 0:
                    return p.X
                else:
                    return 0.0
            self.debug_log_file.close()    
            print("Gurobi model status", self.gmdl.status)
            print("The optimization failed\n")
            print("Computing computeIIS")
            self.gmdl.computeIIS()
            print("Computing computeIIS finished")            
            self.gmdl.write("model.ilp")
            self.debug_log_file.close()
            return NotImplementedError

    def create_input_constraints(self):
        # the uap perturbation.
        if len(self.xs) <= 0:
            return
        delta = self.gmdl.addMVar(self.xs[0].shape[0], lb = -self.eps, ub = self.eps, vtype=grb.GRB.CONTINUOUS, name='uap_delta')
        vs = [self.gmdl.addMVar(self.xs[i].shape[0], lb = self.xs[i].detach().numpy() - self.eps, ub = self.xs[i].detach().numpy() + self.eps, vtype=grb.GRB.CONTINUOUS, name=f'input_{i}') for i in range(self.batch_size)]
        # ensure all inputs are perturbed by the same uap delta.
        for i, v in enumerate(vs):
            self.gmdl.addConstr(v == self.xs[i].detach().numpy() + delta)
        # ds = [[self.gmdl.addMVar(self.xs[i].shape[0], lb= self.xs[i].detach().numpy() - self.xs[j].detach().numpy()-self.tolerence, ub=self.xs[i].detach().numpy() - self.xs[j].detach().numpy()+self.tolerence, vtype=grb.GRB.CONTINUOUS, name=f'input({i}-{j})') for j in range(i+1, self.batch_size)] for i in range(self.batch_size)]
        # for i in range(self.batch_size):
        #     for j in range(i+1, self.batch_size):
        #         self.gmdl.addConstr(ds[i][j - i -1] == self.xs[i].detach().numpy() - self.xs[j].detach().numpy())                
        
        self.gurobi_variables.append({'delta': delta, 'vs': vs, 'ds': None})

    def create_constraints(self):
        self.create_input_constraints()
        layers = self.mdl
        for layer_idx, layer in enumerate(layers):
            layer_type = self.get_layer_type(layer)
            if layer_type == LayerType.Linear:
                self.linear_layer_idx += 1
                self.create_linear_constraints(layer, layer_idx)
            elif layer_type == LayerType.ReLU:
                self.create_relu_constraints(layer_idx)
            elif layer_type == LayerType.Conv2D:
                self.linear_layer_idx += 1                
                self.create_conv2d_constraints(layer, layer_idx)
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")

    def create_vars(self, layer_idx, layer_type=''):
        if layer_type in ['linear', 'conv2d']:            
            vs = [self.gmdl.addMVar(self.x_lbs[i][self.linear_layer_idx].shape[0], lb = self.x_lbs[i][self.linear_layer_idx], ub = self.x_ubs[i][self.linear_layer_idx], vtype=grb.GRB.CONTINUOUS, name=f'layer_{layer_idx}_{layer_type}_x{i}') for i in range(self.batch_size)]
            ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][self.linear_layer_idx].shape[0], lb=self.d_lbs[(i, j)][self.linear_layer_idx] - self.tolerence, ub=self.d_ubs[(i, j)][self.linear_layer_idx] + self.tolerence, vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') for j in range(i+1, self.batch_size)] for i in range(self.batch_size)]
        elif layer_type == 'relu':
            vs = [self.gmdl.addMVar(self.x_lbs[i][self.linear_layer_idx].shape[0], lb =np.maximum(self.x_lbs[i][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0])),
                                     ub = np.maximum(self.x_ubs[i][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0])), vtype=grb.GRB.CONTINUOUS, name=f'layer_{layer_idx}_{layer_type}_x{i}') for i in range(self.batch_size)]
            ds = [[self.gmdl.addMVar(self.d_lbs[(i, j)][self.linear_layer_idx].shape[0], lb=np.maximum(self.x_lbs[i][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0])) - np.maximum(self.x_ubs[j][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0])),
                                     ub=np.maximum(self.x_ubs[i][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0])) - np.maximum(self.x_lbs[j][self.linear_layer_idx], np.zeros(self.x_ubs[i][self.linear_layer_idx].shape[0])),
                                      vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') for j in range(i+1, self.batch_size)] for i in range(self.batch_size)]
        else:
            raise ValueError(f'layer type {layer_type} is supported yet')

        return vs, ds
    
    def create_linear_constraints(self, layer, layer_idx):
        weight, bias = layer.weight, layer.bias
       
        weight = weight.detach().numpy()
        bias = bias.detach().numpy()
        vs, ds = self.create_vars(layer_idx, 'linear')
        
        for v_idx, v in enumerate(vs):
            self.gmdl.addConstr(v == weight @ self.gurobi_variables[-1]['vs'][v_idx] + bias)

        # if self.debug:
        #     self.gmdl.setObjective(vs[0][1], grb.GRB.MINIMIZE)
        #     self.gmdl.optimize()
        #     v_lb = vs[0][1].X
        #     base_lb = self.x_lbs[0][self.linear_layer_idx][1]
        #     base_ub = self.x_ubs[0][self.linear_layer_idx][1]            
        #     self.gmdl.setObjective(vs[0][1], grb.GRB.MAXIMIZE)
        #     self.gmdl.optimize()
        #     v_ub = vs[0][1].X
        #     self.debug_log_file.write(f"lb : {v_lb} ub : {v_ub}\n\n")
        #     self.debug_log_file.write(f"baseline lb  : {base_lb} ub : {base_ub}\n\n")
        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(vs[i] - vs[j] <= (self.d_ubs[(i, j)][self.linear_layer_idx].detach().numpy() + self.tolerence))
                    self.gmdl.addConstr(vs[i] - vs[j] >= (self.d_lbs[(i, j)][self.linear_layer_idx].detach().numpy() - self.tolerence))

            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])

        # for i in range(self.batch_size):
        #     for j in range(i+1, self.batch_size):
        #         if self.linear_layer_idx > 0:
        #             self.gmdl.addConstr(ds[i][j - i - 1] <= weight @ self.gurobi_variables[-1]['ds'][i][j - i -1] + self.tolerence)
        #             self.gmdl.addConstr(ds[i][j - i - 1] >= weight @ self.gurobi_variables[-1]['ds'][i][j - i -1] - self.tolerence)
        #         self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])        
        self.gurobi_variables.append({'vs': vs, 'ds': ds})


    def create_relu_ub(self, x, lb, ub): #probably breaks on degenerate lb=ub case, should fix
        rlb, rub = np.max(0, lb), np.max(0, ub)
        return (rub-rlb)/(ub-lb) * (x-lb) + rlb

    def create_relu_constraints(self, layer_idx):
        vs, ds = self.create_vars(layer_idx, 'relu')

        for i in range(self.batch_size):
            self.gmdl.addConstr(vs[i] >= 0)
            self.gmdl.addConstr(vs[i] >= self.gurobi_variables[-1]['vs'][i])
            tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0] 
            for j in range(tensor_length):                    
                if self.x_lbs[i][self.linear_layer_idx][j] >= 0:
                    self.gmdl.addConstr(vs[i][j] <= self.gurobi_variables[-1]['vs'][i][j])
                    continue 
                elif self.x_ubs[i][self.linear_layer_idx][j] <= 0:
                    self.gmdl.addConstr(vs[i][j] <= 0)
                    continue
                else:
                    ub = self.x_ubs[i][self.linear_layer_idx][j]
                    lb = self.x_lbs[i][self.linear_layer_idx][j]
                    slope = ub / (ub - lb + 1e-15)
                    mu = -slope * lb
                    self.gmdl.addConstr(vs[i][j] <= slope * self.gurobi_variables[-1]['vs'][i][j] + mu)
        
        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])

            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    tensor_length = self.x_lbs[i][self.linear_layer_idx].shape[0]
                    for k in range(tensor_length):
                        # case 1 x unsettled & y passive
                        x_active = (self.x_lbs[i][self.linear_layer_idx][k] >= 0)
                        x_passive = (self.x_ubs[i][self.linear_layer_idx][k] <= 0)
                        x_unsettled = (~x_active) & (~x_passive)
                        y_active = (self.x_lbs[j][self.linear_layer_idx][k] >= 0)
                        y_passive = (self.x_ubs[j][self.linear_layer_idx][k] <= 0)
                        y_unsettled = (~y_active) & (~y_passive)
                        delta_active = (self.d_lbs[(i, j)][self.linear_layer_idx][k] >= 0)
                        delta_passive = (self.d_ubs[(i, j)][self.linear_layer_idx][k] <= 0)
                        delta_unsettled = (~delta_active) & (~delta_passive)                                                                                
                        if x_unsettled and y_passive and delta_active:
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        elif x_unsettled and y_active:
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        elif x_passive and y_unsettled and delta_passive:
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        elif x_active and y_unsettled:
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                        elif x_unsettled and y_unsettled and delta_active:
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= 0.0)                        
                        elif x_unsettled and y_unsettled and delta_passive:
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= self.gurobi_variables[-1]['ds'][i][j - i -1][k])
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= 0.0)
                        elif x_unsettled and y_unsettled and delta_unsettled:
                            temp_lb = self.d_lbs[(i, j)][self.linear_layer_idx][k]
                            temp_ub = self.d_ubs[(i, j)][self.linear_layer_idx][k]
                            d_lambda_lb = temp_lb / (temp_lb - temp_ub + 1e-15)
                            d_lambda_ub = temp_ub / (temp_ub - temp_lb + 1e-15)
                            d_mu_lb = d_lambda_ub * temp_lb
                            d_mu_ub = -d_lambda_ub * temp_lb
                            self.gmdl.addConstr(ds[i][j - i -1][k] >= d_lambda_lb * self.gurobi_variables[-1]['ds'][i][j - i -1][k] + d_mu_lb)
                            self.gmdl.addConstr(ds[i][j - i -1][k] <= d_lambda_ub * self.gurobi_variables[-1]['ds'][i][j - i -1][k] + d_mu_ub)                                                
                                    
        self.gurobi_variables.append({'vs': vs, 'ds': ds})

    
    def create_conv2d_constraints_helper(self, var, pre_var, num_kernel, output_h, 
                                         output_w, bias, weight, layer, input_h, input_w):
        out_idx = 0
        for out_chan_idx in range(num_kernel):
            for out_row_idx in range(output_h):
                for out_col_idx in range(output_w):
                    lin_expr = bias[out_chan_idx].item()

                    for in_chan_idx in range(weight.shape[1]):
                        for ker_row_idx in range(weight.shape[2]):
                            in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                            if (in_row_idx < 0) or (in_row_idx == input_h):
                                # This is padding -> value of 0
                                continue
                            for ker_col_idx in range(weight.shape[3]):
                                in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                if (in_col_idx < 0) or (in_col_idx == input_w):
                                    # This is padding -> value of 0
                                    continue
                                coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()

                                lin_expr += coeff * pre_var[in_chan_idx * (input_h * input_w) + in_row_idx * (input_w) + in_col_idx]
                    self.gmdl.addConstr(var[out_idx] == lin_expr)
                    out_idx += 1


        
    def create_conv2d_constraints(self, layer, layer_idx):
        vs, ds = self.create_vars(layer_idx, 'conv2d')
        weight = layer.weight
        bias = layer.bias
        assert layer.dilation == (1, 1)

        # ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding
        num_kernel = weight.shape[0]
        input_h, input_w = self.shape[1:]
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        # Updated shape
        self.shape = (num_kernel, output_h, output_w)
        for v_idx, v in enumerate(vs):
            self.create_conv2d_constraints_helper(var=v, pre_var=self.gurobi_variables[-1]['vs'][v_idx],
                                                  num_kernel=num_kernel, output_h=output_h, output_w=output_w,
                                                  bias=bias, weight=weight, layer=layer, input_h=input_h, input_w=input_w)
        
        if self.track_differences is True:
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(vs[i] - vs[j] <= (self.d_ubs[(i, j)][self.linear_layer_idx].detach().numpy() + self.tolerence))
                    self.gmdl.addConstr(vs[i] - vs[j] >= (self.d_lbs[(i, j)][self.linear_layer_idx].detach().numpy() - self.tolerence))

            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    self.gmdl.addConstr(ds[i][j - i - 1] == vs[i] - vs[j])
                
        self.gurobi_variables.append({'vs': vs, 'ds': ds})

    def get_layer_type(self, layer):
        return layer.type
