import gurobipy as grb
import torch
from torch import nn
from src.common.network import Network, LayerType
import numpy as np

class UAPLPtransformer:
    def __init__(self, mdl, xs, init_r, x_lbs, x_ubs, d_lbs, d_ubs):
        self.mdl = mdl
        self.xs = xs
        self.batch_size = len(xs)
        self.init_r = init_r
        self.x_lbs = x_lbs
        self.x_ubs = x_ubs
        self.d_lbs = d_lbs
        self.d_ubs = d_ubs
        self.d_idx = [[i for i in range(len(d_lbs)) if d_lbs[i] is not None] for _ in range(len(d_lbs[0]))]

        self.gmdl = grb.Model()
        self.gurobi_variables = []

    def create_lp(self):
        self.gmdl.setParam('OutputFlag', False)
        self.create_constraints()

    def optimize_lp(self, constraint_matrix):
        constraints = [constraint_matrix @ self.gurobi_variables[-1]['vs'][i] for i in range(self.batch_size)]
        ts = [self.gmdl.addMVar(constraints[i].shape, vtype=grb.GRB.CONTINUOUS, name=f't{i}') for i in range(self.batch_size)]
        t_mins = []
        for idx, t in enumerate(ts):
            self.gmdl.addConstr(t == constraints[idx])
            t_mins.append(self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f't_min{idx}'))
            self.gmdl.addGenConstrMin(t_mins[-1], t.tolist())
        bs = [self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f'b{i}') for i in range(self.batch_size)]
        
        t_max = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f't_max')
        self.gmdl.addGenConstrMax(t_max, t_mins)
        self.gmdl.reset()
        self.gmdl.setObjective(t_max, grb.GRB.MINIMIZE)
        self.gmdl.optimize()
        if self.gmdl.status == 2:
            return t_max.X
        else:
            return NotImplementedError
    
    def optimize_milp_percent(self, constraint_matrix):
        constraints = [constraint_matrix @ self.gurobi_variables[-1]['vs'][i] for i in range(self.batch_size)]
        ts = [self.gmdl.addMVar(constraints[i].shape, vtype=grb.GRB.CONTINUOUS, name=f't{i}') for i in range(self.batch_size)]
        bs = []
        for idx, t in enumerate(ts):
            self.gmdl.addConstr(t == constraints[idx])
            t_min = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f't_min{idx}')
            self.gmdl.addGenConstrMin(t_min, t.tolist())
            bs.append(self.gmdl.addVar(vtype=grb.GRB.BINARY, name=f'b{idx}'))
            self.gmdl.addConstr(bs[-1] == (t_min >= 0))
        p = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f'p')
        self.gmdl.addConstr(p == self.gmdl.quicksum(bs[i] for i in range(self.batch_size)) / self.batch_size)
        self.gmdl.reset()
        self.gmdl.setObjective(p, grb.GRB.MINIMIZE)
        self.gmdl.optimize()
        if self.gmdl.status == 2:
            return p.X
        else:
            return NotImplementedError

    def create_input_constraints(self):
        vs = [self.gmdl.addMVar(self.xs[i].shape, lb = self.xs[i] - self.init_r, ub = self.xs[i] + self.init_r, vtype=grb.GRB.CONTINUOUS, name=f'input') for i in range(self.batch_size)]
        self.gurobi_variables.append({'vs': vs, 'ds': None})

    def create_constraints(self):
        self.create_input_constraints()
        layers = self.mdl
        for layer_idx, layer in enumerate(layers):
            layer_type = self.get_layer_type(layer)

            if layer_type == LayerType.Linear:
                self.create_linear_constraints(layer, layer_idx)
            elif layer_type == LayerType.ReLU:
                self.create_relu_constraints(layer_idx)
            elif layer_type == LayerType.Conv2D:
                self.create_conv2d_constraints(layer, layer_idx)
            elif layer_type == LayerType.Flatten:
                continue
            else:
                raise TypeError(f"Unsupported Layer Type '{layer_type}'")

    def create_vars(self, layer_idx):
        vs = [self.gmdl.addMVar(self.x_lbs[layer_idx][i].shape, lb = self.x_lbs[layer_idx][i], ub = self.x_ubs[layer_idx][i], vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_x{i}') for i in range(self.batch_size)]
        ds = [[self.gmdl.addMVar(self.d_lbs[layer_idx][i][j].shape, lb = self.d_lbs[layer_idx][i][j], ub = self.d_ubs[layer_idx][i][j], vtype=grb.GRB.CONTINUOUS, name=f'layer{layer_idx}_d({i}-{j})') for j in self.d_idx[i]] for i in range(len(self.d_idx))]
        return vs, ds

    def create_linear_constraints(self, layer, layer_idx):
        weight, bias = layer.weight, layer.bias

        vs, ds = self.create_vars(layer_idx)

        for v_idx, v in enumerate(vs):
            self.gmdl.addConstr(v == weight @ self.gurobi_variables[-1]['vs'][v_idx] + bias)
        for i in range(len(self.d_idx)):
            for k, j in enumerate(self.d_idx[i]):
                self.gmdl.addConstr(self.gurobi_variables[-1]['ds'][i][k] == weight @ (self.gurobi_variables[-1]['vs'][i] - self.gurobi_variables[-1]['vs'][j])) #is this right?

        self.gurobi_variables.append({'vs': vs, 'ds': ds})

    def create_relu_ub(self, x, lb, ub): #probably breaks on degenerate lb=ub case, should fix
        rlb, rub = np.max(0, lb), np.max(0, ub)
        return (rub-rlb)/(ub-lb) * (x-lb) + rlb

    def create_relu_constraints(self, layer_idx):
        vs, ds = self.create_vars(layer_idx)

        for i in range(self.batch_size):
            self.gmdl.addConstr(vs[i] >= 0)
            self.gmdl.addConstr(vs[i] >= self.gurobi_variables[-1]['vs'][i])
            self.gmdl.addConstr(vs[i] <= self.create_relu_ub(self.gurobi_variables[-1]['vs'][i], self.x_lbs[layer_idx-1][i], self.x_ubs[layer_idx-1][i]))
        
        for k in range(len(ds)): #this seems terrible and slow, ill speed it up
            for l in range(len(ds[k])):
                d = ds[k][l].tolist()
                r = self.d_idx[k][l]
                for i in range(len(d)):
                    for j in range(len(d[0])): #check constraints
                        if self.x_ubs[layer_idx][k][i][j] <= 0 and self.x_ubs[layer_idx][r][i][j] <= 0:
                            self.gmdl.addConstr(d[i][j] == 0)
                        #elif self.x1_u[layer_idx][i][j] <= 0 and self.x2_l[layer_idx][i][j] >= 0:
                        #    self.gmdl.addConstr(d[i][j] == -self.gurobi_variables[-1]['v2'])
                        #elif self.x1_l[layer_idx][i][j] >= 0 and self.x2_u[layer_idx][i][j] <= 0:
                        #    self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['v1'])
                        elif self.x_lbs[layer_idx][k][i][j] >= 0 and self.x_lbs[layer_idx][r][i][j] >= 0:
                            self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['vs'][k] - self.gurobi_variables[-1]['vs']['r'])
                        elif self.x_ubs[layer_idx][k][i][j] <= 0:
                            self.gmdl.addConstr(d[i][j] == -self.gurobi_variables[-1]['vs'][r])
                        #elif self.x1_l[layer_idx][i][j] >= 0:
                        elif self.x_ubs[layer_idx][r][i][j] <= 0:
                            self.gmdl.addConstr(d[i][j] == self.gurobi_variables[-1]['vs'][k])
                        #elif self.x2_l[layer_idx][i][j] >= 0:
                        else:
                            self.gmdl.addConstr(d[i][j] >= 0)
                            self.gmdl.addConstr(d[i][j] >= self.gurobi_variables[-1]['vs'][k] - self.gurobi_variables[-1]['vs'][r])
                            self.gmdl.addConstr(d[i][j] <= self.create_relu_ub(self.gurobi_variables[-1]['vs'][k] - self.gurobi_variables[-1]['vs'][r], self.d_lbs[layer_idx-1][i][j], self.d_ubs[layer_idx-1][i][j])) #not sure
                            
        self.gurobi_variables.append({'vs': vs, 'ds': ds})
        
    def create_conv2d_constraints(self, layer, layer_idx):
        raise NotImplementedError

    def get_layer_type(self, layer):
        return layer.type

    # def get_layer_type(self, layer):
    #     if self.format == "onnx":
    #         return layer.type

    #     if self.format == "torch":
    #         if type(layer) is nn.Linear:
    #             return LayerType.Linear
    #         elif type(layer) is nn.Conv2d:
    #             return LayerType.Conv2D
    #         elif type(layer) == nn.ReLU:
    #             return LayerType.ReLU
    #         elif type(layer) == nn.Flatten():
    #             return LayerType.Flatten
    #         else:
    #             return LayerType.NoOp
    #             # raise ValueError("Unsupported layer type for torch model!", type(layer))

    #     raise ValueError("Unsupported model format or model format not set!")