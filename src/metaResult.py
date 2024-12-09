from collections import defaultdict
class PropertyResult:
    def __init__(self):
        self.verified_dict = {}
        self.raven_args = None 
        self.alg_results = defaultdict(dict)
        self.incomplete_tuples = None
        self.incomplete_losses = None 
        self.complete_tuples = None 
        self.complete_losses = None 
        self.targetted_indices = None 
        self.meta_acc = None 
        self.abc_acc = None 
        self.meta_topk = []
        self.abc_topk = []
        self.unverified_indices = None
        

class OverallResult:
    def __init__(self):
        self.results_lst = []
    
    def add_res(self, res):
        self.results_lst.append(res)
    
    