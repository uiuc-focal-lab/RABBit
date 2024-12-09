from src.common import Status

class UAPSingleRes:
    def __init__(self, domain, input_per_prop, status, global_lb, time_taken, verified_proportion):
        self.domain = domain
        self.status = status
        self.input_per_prop = input_per_prop
        self.global_lb = global_lb
        self.time_taken = time_taken
        self.verified_proportion = verified_proportion

    def print(self):
        print("Domain ", self.domain)
        print("Time taken ", self.time_taken)
        print("Global lb ", self.global_lb)
        print("Status ", self.status)
        print("Input per prop ", self.input_per_prop)
        print("Verified proportion", self.verified_proportion)

class UAPResult:
    def __init__(self, UAP_res, baseline_res):
        self.baseline_res = baseline_res
        self.UAP_res = UAP_res

class UAPResultList:
    def __init__(self) -> None:
        self.result_list = []

    def add_results(self, res: UAPResult):
        self.result_list.append(res)

    def analyze(self, args):
        count = args.count
        baseline_verified_count = 0
        uap_verified_count = 0        
        filename = args.output_dir + f'{args.net_name}_{args.count_per_prop}_{args.count}_{args.eps}.dat'
        file = open(filename, 'a+')
        for i, res in enumerate(self.result_list):
            baseline_res = res.baseline_res
            UAP_res = res.UAP_res
            file.write(f'\nProperty No. {i}\n\n')
            if baseline_res.verified_proportion is not None:
                baseline_verified_count += baseline_res.verified_proportion * args.count_per_prop
                file.write(f"baseline verified proportion {baseline_res.verified_proportion}\n")

            if UAP_res.verified_proportion is not None:
                uap_verified_count += UAP_res.verified_proportion * args.count_per_prop
                file.write(f"Uap verified proportion {UAP_res.verified_proportion}\n")
        file.write(f'\n\n\nEps : {args.eps}\n')
        file.write(f'Baseline verified: {baseline_verified_count}\n')
        file.write(f'Uap verified: {uap_verified_count}\n')
        file.write(f'Extra verified: {uap_verified_count - baseline_verified_count}\n')
        file.close()                 