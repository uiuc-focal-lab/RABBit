from src.common import Dataset
import torch


class Result:
    def __init__(self, final_result, final_time,
                result_trace=None, time_trace=None, lp_result=None, 
                lp_time=None) -> None:
        self.final_result = final_result
        self.final_time = final_time
        self.result_trace = result_trace
        self.time_trace = time_trace
        self.lp_result = lp_result
        self.lp_time = lp_time

class AdaptiveRavenResult:
    def __init__(self, individual_res : Result, baseline_res : Result, 
                 individual_refinement_res : Result, 
                 individual_refinement_milp_res : Result,
                 cross_executional_refinement_res : Result, 
                 final_res : Result) -> None:
        self.individual_res = individual_res
        self.baseline_res = baseline_res
        self.individual_refinement_res = individual_refinement_res
        self.individual_refinement_milp_res = individual_refinement_milp_res
        self.cross_executional_refinement_res = cross_executional_refinement_res
        self.final_res = final_res

class AdaptiveRavenResultList:
    def __init__(self, args) -> None:
        self.res_list = []
        self.args = args
        self.individual_refinement_bounds_trace = {}
        self.individual_refinement_time_trace = {}
        self.individual_refinement_lp_time = {}
        self.cross_ex_bounds_trace = {}
        self.cross_ex_time_trace = {}
        self.individual_lp_bound = {}
        self.cross_ex_lp_bound = {}
        self.cross_ex_lp_time = {}

    def populate_trace(self, execution_count, individual_bound_trace, individual_time_trace,
                                cross_ex_bounds_trace, cross_ex_time_trace, 
                                individual_lp_bound, individual_lp_time,
                                cross_ex_lp_bound, cross_ex_lp_time):
        self.individual_refinement_bounds_trace[execution_count] = individual_bound_trace
        self.individual_refinement_time_trace[execution_count] = individual_time_trace
        self.cross_ex_bounds_trace[execution_count] = cross_ex_bounds_trace
        self.cross_ex_time_trace[execution_count] = cross_ex_time_trace
        self.individual_lp_bound[execution_count] = individual_lp_bound
        self.cross_ex_lp_bound[execution_count] = cross_ex_lp_bound
        self.individual_refinement_lp_time[execution_count] = individual_lp_time
        self.cross_ex_lp_time[execution_count]=cross_ex_lp_time

    def populate_or_replace_trace(self, execution_count, individual_bound_trace, individual_time_trace,
                                cross_ex_bounds_trace, cross_ex_time_trace, 
                                individual_lp_bound, individual_lp_time,
                                cross_ex_lp_bound, cross_ex_lp_time):
        if execution_count not in self.individual_refinement_bounds_trace.keys():
            self.populate_trace(execution_count, individual_bound_trace, individual_time_trace,
                                cross_ex_bounds_trace, cross_ex_time_trace, 
                                individual_lp_bound, individual_lp_time,
                                cross_ex_lp_bound, cross_ex_lp_time)
            return
        if self.individual_lp_bound[execution_count] >= 0.0 and individual_lp_bound < 0.0:
            self.populate_trace(execution_count, individual_bound_trace, individual_time_trace,
                                cross_ex_bounds_trace, cross_ex_time_trace, 
                                individual_lp_bound, individual_lp_time,
                                cross_ex_lp_bound, cross_ex_lp_time)
            return

        if self.individual_lp_bound[execution_count] < 0.0:
            return
        
        prev_bound_diff = self.cross_ex_bounds_trace[execution_count][-1] - self.individual_refinement_bounds_trace[execution_count][-1]
        if (cross_ex_bounds_trace[-1] - individual_bound_trace[-1]) > prev_bound_diff:
            self.populate_trace(execution_count, individual_bound_trace, individual_time_trace,
                                cross_ex_bounds_trace, cross_ex_time_trace, 
                                individual_lp_bound, individual_lp_time,
                                cross_ex_lp_bound, cross_ex_lp_time)
            return


    def add_res(self, res : AdaptiveRavenResult):
        self.res_list.append(res)
    
    def get_file(self):
        eps = self.args.eps if self.args.dataset == Dataset.MNIST else self.args.eps * 255
        filename = self.args.result_dir + '/' + f'{self.args.net_names[0]}_{self.args.count_per_prop}_{eps}.dat'
        file = open(filename, 'a+')
        return file

    def bound_file(self, ex_count):
        eps = self.args.eps if self.args.dataset == Dataset.MNIST else self.args.eps * 255
        filename = 'bounds_ablation' + '/' + f'{self.args.net_names[0]}_{ex_count}_{eps}.dat'
        file = open(filename, 'a+')
        return file

    def write_bounds(self):

        for ex_count in self.individual_refinement_bounds_trace.keys():
            if ex_count not in self.cross_ex_bounds_trace.keys():
                continue
            file = self.bound_file(ex_count=ex_count)
            
            indiv_bnd_trace = self.individual_refinement_bounds_trace[ex_count]
            # print(f'inv bnd trace {indiv_bnd_trace}')
            cross_ex_bnd_trace = self.cross_ex_bounds_trace[ex_count]
            # print(f'cross ex bnd trace {cross_ex_bnd_trace}')
            indiv_time_trace = self.individual_refinement_time_trace[ex_count]
            cross_ex_time_trace = self.cross_ex_time_trace[ex_count]
            length = max(indiv_bnd_trace.shape[0], cross_ex_bnd_trace.shape[0])
            for i in range(length):
                inv_bnd = indiv_bnd_trace[min(i, indiv_bnd_trace.shape[0] -1)]
                cross_ex_bnd = cross_ex_bnd_trace[min(i, cross_ex_bnd_trace.shape[0] -1)]
                if type(inv_bnd) is torch.Tensor:
                    inv_bnd = inv_bnd.item()
                if type(cross_ex_bnd) is torch.Tensor:
                    cross_ex_bnd = cross_ex_bnd.item()
                file.write(f'bounds: {inv_bnd} {cross_ex_bnd}\n') 

            file.write(f'lp_bound: {self.individual_lp_bound[ex_count]} {self.cross_ex_lp_bound[ex_count]}\n')
            file.write(f'lp_time: {self.individual_refinement_lp_time[ex_count]} {self.cross_ex_lp_time[ex_count]}\n')        
            file.close()

    def bounds_comparsion(self):
        for res in self.res_list:
            cross_ex_res = res.cross_executional_refinement_res
            indivudiual_res = res.individual_refinement_res
            if cross_ex_res.result_trace is None or indivudiual_res.result_trace is None:
                continue
            for ex_count in cross_ex_res.result_trace.keys():
                if ex_count not in indivudiual_res.result_trace.keys():
                    continue
                self.populate_or_replace_trace(execution_count=ex_count,
                                            individual_bound_trace=indivudiual_res.result_trace[ex_count],
                                            individual_time_trace=indivudiual_res.time_trace[ex_count],
                                            cross_ex_bounds_trace=cross_ex_res.result_trace[ex_count],
                                            cross_ex_time_trace=cross_ex_res.time_trace[ex_count], 
                                            individual_lp_bound=indivudiual_res.lp_result[ex_count],
                                            individual_lp_time=indivudiual_res.lp_time[ex_count],
                                            cross_ex_lp_bound=cross_ex_res.lp_result[ex_count],
                                            cross_ex_lp_time=cross_ex_res.lp_time[ex_count])
        self.write_bounds()
        


    def analyze(self):
        individual_acc = 0
        baseline_acc = 0
        individual_refinement_acc = 0
        individual_refinement_milp_acc = 0
        cross_ex_acc = 0
        final_acc = 0

        individual_time = 0
        baseline_time = 0
        individual_refinement_time = 0
        individual_refinement_milp_time = 0
        cross_ex_time = 0
        final_time = 0
        count = 0
        
        def populate(res : Result, acc, time):
            if res.final_result is not None:
                acc += res.final_result
            if res.final_time is not None:
                time += res.final_time
            return acc, time

        for res in self.res_list:
            individual_acc, individual_time = populate(res.individual_res, individual_acc, individual_time)
            baseline_acc, baseline_time = populate(res.baseline_res, baseline_acc, baseline_time)
            individual_refinement_acc, individual_refinement_time = populate(res.individual_refinement_res, 
                                                                             individual_refinement_acc, individual_refinement_time)
            
            individual_refinement_milp_acc, individual_refinement_milp_time = populate(res.individual_refinement_milp_res, 
                                                                             individual_refinement_milp_acc, 
                                                                             individual_refinement_milp_time)
            cross_ex_acc, cross_ex_time = populate(res.cross_executional_refinement_res, cross_ex_acc, cross_ex_time)
            final_acc, final_time = populate(res.final_res, final_acc, final_time)
        assert len(self.res_list) > 0
        scale = 1/ len(self.res_list)
        if self.args.write_file:
            file = self.get_file()
            file.write(f'individual acc {individual_acc*scale}\n')
            file.write(f'baseline acc {baseline_acc*scale}\n')
            file.write(f'IndivRefine acc {individual_refinement_acc*scale}\n')
            file.write(f'IndivRefineMILP acc {individual_refinement_milp_acc*scale}\n')            
            file.write(f'CrossEx acc {cross_ex_acc*scale}\n')
            file.write(f'final acc {final_acc*scale}\n')

            file.write(f'individual time {individual_time*scale}\n')
            file.write(f'baseline time {baseline_time*scale}\n')
            file.write(f'IndivRefine time {individual_refinement_time*scale}\n')
            file.write(f'IndivRefineMILP time {individual_refinement_milp_time*scale}\n')
            file.write(f'CrossEx time {cross_ex_time*scale}\n')        
            file.write(f'final time {final_time*scale}\n')

            file.close()
        if self.args.populate_trace:
            self.bounds_comparsion()