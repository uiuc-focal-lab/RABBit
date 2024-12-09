import torch
import numpy as np
import torch.nn.functional as F
from src.specLoader import get_specification, get_std
from src.netLoader import get_net
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode
from raven.src.network_conversion_helper import get_pytorch_net

def project_lp(v, norm, xi, exact = False, device = 'cpu'):
    if v.dim() == 4:
        batch_size = v.shape[0]
    else:
        batch_size = 1
    if exact:
        if norm == 2:
            if batch_size == 1:
                v = v * xi/torch.norm(v, p = 2)
            else:
                v = v * xi/torch.norm(v, p = 2, dim = (1,2,3)).reshape((batch_size, 1, 1, 1))
        elif norm == np.inf:        
            v = torch.sign(v) * torch.minimum(torch.abs(v), xi*torch.ones(v.shape, device = device))
        else:
            raise ValueError('L_{} norm not implemented'.format(norm))
    else:
        if norm == 2:
            if batch_size == 1:
                v = v * torch.minimum(torch.ones((1), device = device), xi/torch.norm(v, p = 2))
            else:
                v = v * torch.minimum(xi/torch.norm(v, p = 2, dim = (1,2,3)), torch.ones(batch_size, device = device)).reshape((batch_size, 1, 1, 1))
        elif norm == np.inf:        
            v = torch.sign(v) * torch.minimum(torch.abs(v), xi*torch.ones(v.shape, device = device))
        else:
            raise ValueError('L_{} norm not implemented'.format(norm))
    return v

# Returns the lower bounds of different_executions = batch_size // execution_count 
def run_uap_attack(model, inputs, constraint_matrices,
                   eps, execution_count, epochs, restarts):
    if len(inputs.shape) < 4:
        raise ValueError("We only support batched inputs")
    assert inputs.shape[0] == constraint_matrices.shape[0]
    assert inputs.shape[0] % execution_count == 0
    if type(eps) is torch.Tensor:
        eps = eps.min()
    device = inputs.device
    different_execution = inputs.shape[0] // execution_count
    final_min_attack_loss = None
    for _ in range(restarts):
        random_delta = torch.rand(different_execution, *inputs.shape[1:], device = device) - 0.5
        random_delta = project_lp(random_delta, norm =np.inf, xi = eps, exact = True, device = device)
        for j in range(epochs):
            # random_delta.requires_grad = True
            indices = torch.arange(end=different_execution, device=device).repeat(execution_count)
            # print(f"Indices {indices}")
            pert_x = inputs + random_delta[indices]
            pert_x.requires_grad = True
            output = model(pert_x)
            tranformed_output = torch.stack([constraint_matrices[i].matmul(output[i]) for i in range(inputs.shape[0])])
            tranformed_output_min = tranformed_output.min(dim=1)[0]
            tranformed_output_min = tranformed_output_min.view(execution_count, -1)
            final_output = tranformed_output_min
            final_output = tranformed_output_min.max(dim=0)[0]
            final_min_attack_loss = final_output if final_min_attack_loss is None else torch.min(final_output, final_min_attack_loss)
            # print(f"final output {final_output}")
            loss = final_output.sum()
            loss.backward()
            projected_gradient = pert_x.grad.reshape(execution_count, -1, *pert_x.grad.shape[1:]).mean(dim=0)
            pert = 0.001 * torch.sign(projected_gradient)
            random_delta = project_lp(random_delta - pert, norm = np.inf, xi = eps)
    print(f"Minimum final loss {final_min_attack_loss}")

# def compute_pgd(x, y, k, norm = np.inf, xi = 10, epochs = 40, random_restart = 4, step_size = 1e-2, device = device):
#     batch_size = x.shape[0]
#     max_loss = F.cross_entropy(k(x), y)
#     max_X = torch.zeros_like(x)
#     random_delta = torch.rand(size = (batch_size * random_restart, *x.shape[1:]), device = device) - 0.5
#     random_delta = project_lp(random_delta, norm = norm, xi = xi, exact = True, device = device)
#     x = x.repeat(random_restart, 1, 1, 1)
#     y = y.repeat(random_restart)
#     for j in range(epochs):
#         pert_x = x + random_delta
#         pert_x.requires_grad = True
#         loss = F.cross_entropy(k(pert_x), y)
#         loss.backward()
#         pert = step_size * torch.sign(pert_x.grad)
#         random_delta = project_lp(random_delta + pert, norm = norm, xi = xi)
#     _,idx = torch.max(F.cross_entropy(mdl(x + random_delta), y, reduction = 'none').reshape(random_restart, batch_size), axis = 0)
#     return random_delta[idx * batch_size + torch.arange(batch_size, dtype = torch.int64, device = device)]

# def compute_uap(dl, k, adv_func, norm = np.inf, xi = 10, delta = 0.15, max_iter = 50, device = device, verbose = True, **kwargs):
#     image_shape = next(iter(dl))[0][0].shape
#     v = torch.zeros(size = image_shape, device = device)
#     error = 0 #err(dl, v, k)
#     kwargs['norm'] = norm
#     kwargs['xi'] = xi
#     kwargs['device'] = device
#     itr = 0
#     if verbose:
#         print("Iteration {}: Error Rate - {}".format(itr, error))
#     while error <= 1 - delta and itr < max_iter:
#         itr += 1
#         for batch in (tqdm(dl) if verbose else dl):
#             x_i, y_i = batch
#             x_i.requires_grad = True
#             mask = torch.max(k(x_i + v), axis = 1)[1] ==  torch.max(k(x_i), axis = 1)[1]
#             x_i_m, y_i_m = x_i[mask], y_i[mask]
#             r = adv_func(x_i_m + v, y_i_m, k, **kwargs)
#             v = iter_add(v, r, norm = norm, xi = xi)
#             x_i.requires_grad = False
#         error = err(dl, v, k)
#         if verbose:
#             print("Iteration {}: Error Rate - {}".format(itr, error))
#     return v

if __name__ == "__main__":

    net_names = [config.CIFAR_CONV_SMALL]
    eps = 1.0

    raven_args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
        count_per_prop=40, prop_count=1, eps=eps/255,
        threshold_execution=5, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        refine_intermediate_bounds =True, optimize_layers_count=2, 
        bounds_for_individual_refinement=True, dataloading_seed = 0,
        parallelize_executions=False,
        lp_threshold=-0.5,
        max_linear_apprx=4,
        device='cuda:0',
        always_correct_cross_execution = False,
        result_dir='icml_results_new', write_file=True, complete_verification= True, use_lp_bab= False, meta_scheduling= True)    
    nets = get_net(net_names = raven_args.net_names, dataset = raven_args.dataset)
    total_input_count = raven_args.prop_count * raven_args.count_per_prop
    images, labels, constraint_matrices, lbs, ubs = get_specification(dataset=raven_args.dataset,
                                                            raven_mode=raven_args.raven_mode, 
                                                            count=total_input_count, nets=nets, eps=raven_args.eps,
                                                            dataloading_seed=raven_args.dataloading_seed,
                                                            net_names=raven_args.net_names)
    # tensor([ 2,  3,  4,  5, 13, 15, 19, 20, 25, 32, 33, 37, 38])
    print(f"images shape {images.shape}")
    print(f"labels {labels}")
    print(f"constraint matrices {constraint_matrices.shape}")
    eps=raven_args.eps / get_std(dataset=raven_args.dataset, transform=False)
    filtered_indices = torch.tensor([2, 3, 4,  5, 37, 38])
    filtered_images = images[filtered_indices]
    filtered_constraint_matrices = constraint_matrices[filtered_indices]
    model = get_pytorch_net(model=nets[0], remove_last_layer=False, all_linear=False)
    run_uap_attack(model=model, inputs=filtered_images, 
                   constraint_matrices=filtered_constraint_matrices, 
                   eps=eps, execution_count=2, epochs=40, restarts=20)