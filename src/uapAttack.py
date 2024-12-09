import torch
import numpy as np

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

# Returns the lower bounds of different_executions = batch_size // execution_count.
def run_uap_attack(model, inputs, constraint_matrices,
                   eps, execution_count, epochs=40, restarts=20):
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
    return final_min_attack_loss