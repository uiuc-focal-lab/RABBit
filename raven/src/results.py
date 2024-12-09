class UAPSingleRes:
    def __init__(self, domain, input_per_prop, status, global_lb, time_taken):
        self.domain = domain
        self.status = status
        self.input_per_prop = input_per_prop
        self.global_lb = global_lb
        self.time_taken = time_taken

class UAPResult:
    def __init__(self, UAP_res, baseline_res):
        self.baseline_res = baseline_res
        self.UAP_res = UAP_res

class UAPResultList:
    def __init__(self) -> None:
        self.result_list = []

    def add_results(self, res: UAPResult):
        self.result_list.append(res)

class UAPSingleRadiusRes:
    def __init__(self, domain, input_per_prop, radius, time_taken):
        self.domain = domain
        self.input_per_prop = input_per_prop
        self.radius = radius
        self.time_taken = time_taken