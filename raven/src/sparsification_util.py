import torch


def prune_last_layer(weight, indices):
    sz = weight.size()
    for ind in indices:
        if ind < sz[1]:
            with torch.no_grad():
                weight[:, ind] = 0
        else:
            raise ValueError("Inidices out of range")


def get_sparsification_indices(f_lb, f_ub, final_layer_wt,
                            const_mat):
    out_constraint_mat = const_mat.T
    final_wt = out_constraint_mat @ final_layer_wt
    final_wt = torch.abs(final_wt)
    wt_bounds = torch.max(final_wt, dim=0)
    wt_bounds = wt_bounds[0]    
    abs_feature = torch.maximum(torch.abs(f_lb), torch.abs(f_ub))
    greedy_features = torch.mul(abs_feature, wt_bounds)
    sorted_features = torch.sort(greedy_features)
    nonzero_count = torch.count_nonzero(sorted_features[0])
    zero_fetures_indices = sorted_features[1][:-nonzero_count]
    nonzero_fetures_indices = sorted_features[1][-nonzero_count:]
    return nonzero_count, zero_fetures_indices, nonzero_fetures_indices