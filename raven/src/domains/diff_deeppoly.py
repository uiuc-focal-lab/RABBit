import torch
import torch.nn.functional as F
from src.common.network import LayerType
from src.util import compute_input_shapes

class DiffPropStruct:
    def __init__(self) -> None:
        self.delta_lb_coef = None
        self.delta_lb_bias = None
        self.delta_ub_coef = None
        self.delta_ub_bias = None
        self.delta_lb_input1_coef = None
        self.delta_ub_input1_coef = None
        self.delta_lb_input2_coef = None
        self.delta_ub_input2_coef = None
    
    def populate(self, delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias,
                 delta_lb_input1_coef, delta_ub_input1_coef, delta_lb_input2_coef,
                 delta_ub_input2_coef) -> None:
        self.delta_lb_coef = delta_lb_coef
        self.delta_lb_bias = delta_lb_bias
        self.delta_ub_coef = delta_ub_coef
        self.delta_ub_bias = delta_ub_bias
        self.delta_lb_input1_coef = delta_lb_input1_coef
        self.delta_ub_input1_coef = delta_ub_input1_coef
        self.delta_lb_input2_coef = delta_lb_input2_coef
        self.delta_ub_input2_coef = delta_ub_input2_coef
 

class DiffDeepPoly:
    def __init__(self, input1, input2, net, lb_input1, ub_input1, lb_input2, ub_input2, device='') -> None:
        self.input1 = input1
        self.input2 = input2
        if self.input1.shape[0] == 784:
            self.input_shape = (1, 28, 28)
        elif self.input1.shape[0] == 3072:
            self.input_shape = (3, 32, 32)
        elif self.input1.shape[0] == 2:
            # For unitest only
            self.input_shape = (1, 1, 2)
        else:
            raise ValueError(f"Unrecognised input shape {self.input_shape}")
        self.net = net
        self.lb_input1 = lb_input1
        self.ub_input1 = ub_input1
        self.lb_input2 = lb_input2
        self.ub_input2 = ub_input2
        self.shapes = compute_input_shapes(net=self.net, input_shape=self.input_shape)
        self.diff = input1 - input2
        self.linear_conv_layer_indices = []
        self.device = device
        self.diff_log_filename = "./debug_logs/diff_debug_log.txt"
        self.log_file = None

    # Bias cancels out (Ax + b - Ay - b) = A(x - y) = A * delta 
    def handle_linear(self, linear_wt, bias, back_prop_struct):
        self.log_file.write("Seen linear \n\n")
        delta_lb_bias = back_prop_struct.delta_lb_bias + back_prop_struct.delta_lb_input1_coef.matmul(bias)
        delta_lb_bias = delta_lb_bias + back_prop_struct.delta_lb_input2_coef.matmul(bias)
        delta_ub_bias = back_prop_struct.delta_ub_bias + back_prop_struct.delta_ub_input1_coef.matmul(bias)
        delta_ub_bias = delta_ub_bias + back_prop_struct.delta_ub_input2_coef.matmul(bias)
        
        delta_lb_coef = back_prop_struct.delta_lb_coef.matmul(linear_wt)
        delta_ub_coef = back_prop_struct.delta_ub_coef.matmul(linear_wt)
        delta_lb_input1_coef = back_prop_struct.delta_lb_input1_coef.matmul(linear_wt)
        delta_ub_input1_coef = back_prop_struct.delta_ub_input1_coef.matmul(linear_wt)
        delta_lb_input2_coef = back_prop_struct.delta_lb_input2_coef.matmul(linear_wt)
        delta_ub_input2_coef = back_prop_struct.delta_ub_input2_coef.matmul(linear_wt)                     
        back_prop_struct.populate(delta_lb_coef=delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                                    delta_ub_coef=delta_ub_coef, delta_ub_bias=delta_ub_bias,
                                    delta_lb_input1_coef=delta_lb_input1_coef, 
                                    delta_ub_input1_coef=delta_ub_input1_coef,
                                    delta_lb_input2_coef=delta_lb_input2_coef,
                                    delta_ub_input2_coef=delta_ub_input2_coef)

        return back_prop_struct

    # preconv shape is the shape before the convolution is applied.
    # postconv shape is the shape after the convolution is applied.
    # while back prop the delta coef shape [rows, postconv shape after flattening].
    def handle_conv(self, conv_weight, conv_bias, back_prop_struct, preconv_shape, postconv_shape,
                    stride, padding, groups=1, dilation=(1, 1)):
        kernel_hw = conv_weight.shape[-2:]
        h_padding = (preconv_shape[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_hw[0] - 1)) % stride[0]
        w_padding = (preconv_shape[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_hw[1] - 1)) % stride[1]
        output_padding = (h_padding, w_padding)

        coef_shape = back_prop_struct.delta_lb_coef.shape
        delta_lb_coef = back_prop_struct.delta_lb_coef.view((coef_shape[0], *postconv_shape))
        delta_ub_coef = back_prop_struct.delta_ub_coef.view((coef_shape[0], *postconv_shape))
        delta_lb_input1_coef = back_prop_struct.delta_lb_input1_coef.view((coef_shape[0], *postconv_shape))
        delta_ub_input1_coef = back_prop_struct.delta_ub_input1_coef.view((coef_shape[0], *postconv_shape))
        delta_lb_input2_coef = back_prop_struct.delta_lb_input2_coef.view((coef_shape[0], *postconv_shape))
        delta_ub_input2_coef = back_prop_struct.delta_ub_input2_coef.view((coef_shape[0], *postconv_shape))                        
        
        # delta_lb_bias = back_prop_struct.delta_lb_bias + (0 if conv_bias is None else (delta_lb_coef.sum((2, 3)) * conv_bias).sum(1))
        # delta_ub_bias = back_prop_struct.delta_ub_bias + (0 if conv_bias is None else (delta_ub_coef.sum((2, 3)) * conv_bias).sum(1))
        delta_lb_bias = back_prop_struct.delta_lb_bias + (delta_lb_input1_coef.sum((2, 3)) * conv_bias).sum(1)
        delta_lb_bias = delta_lb_bias + (delta_lb_input2_coef.sum((2, 3)) * conv_bias).sum(1)
        delta_ub_bias = back_prop_struct.delta_ub_bias + (delta_ub_input1_coef.sum((2, 3)) * conv_bias).sum(1)
        delta_ub_bias = delta_ub_bias + (delta_ub_input2_coef.sum((2, 3)) * conv_bias).sum(1)


        new_delta_lb_coef = F.conv_transpose2d(delta_lb_coef, conv_weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_delta_ub_coef = F.conv_transpose2d(delta_ub_coef, conv_weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_delta_lb_input1_coef = F.conv_transpose2d(delta_lb_input1_coef, conv_weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_delta_ub_input1_coef = F.conv_transpose2d(delta_ub_input1_coef, conv_weight, None, stride, padding,
                                    output_padding, groups, dilation)
        new_delta_lb_input2_coef = F.conv_transpose2d(delta_lb_input2_coef, conv_weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_delta_ub_input2_coef = F.conv_transpose2d(delta_ub_input2_coef, conv_weight, None, stride, padding,
                                    output_padding, groups, dilation)
                
        new_delta_lb_coef = new_delta_lb_coef.view((coef_shape[0], -1))
        new_delta_ub_coef = new_delta_ub_coef.view((coef_shape[0], -1))
        new_delta_lb_input1_coef = new_delta_lb_input1_coef.view((coef_shape[0], -1))
        new_delta_ub_input1_coef = new_delta_ub_input1_coef.view((coef_shape[0], -1))
        new_delta_lb_input2_coef = new_delta_lb_input2_coef.view((coef_shape[0], -1))
        new_delta_ub_input2_coef = new_delta_ub_input2_coef.view((coef_shape[0], -1))

        back_prop_struct.populate(delta_lb_coef=new_delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                            delta_ub_coef=new_delta_ub_coef, delta_ub_bias=delta_ub_bias,
                            delta_lb_input1_coef=new_delta_lb_input1_coef, 
                            delta_ub_input1_coef=new_delta_ub_input1_coef,
                            delta_lb_input2_coef=new_delta_lb_input2_coef,
                            delta_ub_input2_coef=new_delta_ub_input2_coef)


        return back_prop_struct

    def pos_neg_weight_decomposition(self, coef):
        neg_comp = torch.where(coef < 0, coef, torch.zeros_like(coef, device=self.device))
        pos_comp = torch.where(coef >= 0, coef, torch.zeros_like(coef, device=self.device))
        return neg_comp, pos_comp
    

    def concretize_bounds(self, back_prop_struct, delta_lb_layer, delta_ub_layer,
                          lb_input1_layer, ub_input1_layer, lb_input2_layer, ub_input2_layer):
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_coef)
        neg_comp_lb_input1, pos_comp_lb_input1 = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_input1_coef)
        neg_comp_ub_input1, pos_comp_ub_input1 = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_input1_coef)
        neg_comp_lb_input2, pos_comp_lb_input2 = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_input2_coef)
        neg_comp_ub_input2, pos_comp_ub_input2 = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_input2_coef)

        lb = neg_comp_lb @ delta_ub_layer + pos_comp_lb @ delta_lb_layer + back_prop_struct.delta_lb_bias
        lb = lb + neg_comp_lb_input1 @ ub_input1_layer + pos_comp_lb_input1 @ lb_input1_layer
    
        lb = lb + neg_comp_lb_input2 @ ub_input2_layer + pos_comp_lb_input2 @ lb_input2_layer
    
        ub = neg_comp_ub @ delta_lb_layer + pos_comp_ub @ delta_ub_layer + back_prop_struct.delta_ub_bias
        ub = ub + neg_comp_ub_input1 @ lb_input1_layer + pos_comp_ub_input1 @ ub_input1_layer
        ub = ub + neg_comp_ub_input2 @ lb_input2_layer + pos_comp_ub_input2 @ ub_input2_layer

        return lb, ub


    # Consider cases based on the state of the relu for different propagation.
    def handle_relu(self, back_prop_struct, 
                    lb_input1_layer, ub_input1_layer, 
                    lb_input2_layer, ub_input2_layer,
                    delta_lb_layer, delta_ub_layer):

        input1_active = (lb_input1_layer >= 0)
        input1_passive = (ub_input1_layer <= 0)
        input1_unsettled = ~(input1_active) & ~(input1_passive)

        input2_active = (lb_input2_layer >= 0)
        input2_passive = (ub_input2_layer <= 0)
        input2_unsettled = ~(input2_active) & ~(input2_passive)

        delta_active = (delta_lb_layer >= 0)
        delta_passive = (delta_ub_layer <= 0)
        delta_unsettled = ~(delta_active) & ~(delta_passive)

        lambda_lb = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_lb_input1 = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub_input1 = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_lb_input2 = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub_input2 = torch.zeros(lb_input1_layer.size(), device=self.device)

        lambda_lb_input1_prop = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub_input1_prop = torch.zeros(lb_input1_layer.size(), device=self.device)
        mu_ub_input1_prop = torch.zeros(lb_input1_layer.size(), device=self.device)        
        lambda_lb_input2_prop = torch.zeros(lb_input1_layer.size(), device=self.device)
        lambda_ub_input2_prop = torch.zeros(lb_input1_layer.size(), device=self.device)
        mu_ub_input2_prop = torch.zeros(lb_input1_layer.size(), device=self.device)

        # input1 is active
        lambda_lb_input1_prop = torch.where(input1_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input1_prop)
        lambda_ub_input1_prop = torch.where(input1_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input1_prop)
        # input1 is unsettled
        temp = torch.where(ub_input1_layer < -lb_input1_layer, torch.zeros(lb_input1_layer.size(), device=self.device), torch.ones(lb_input1_layer.size(), device=self.device))
        lambda_lb_input1_prop = torch.where(input1_unsettled, temp, lambda_lb_input1_prop)
        lambda_ub_input1_prop = torch.where(input1_unsettled, ub_input1_layer/(ub_input1_layer - lb_input1_layer + 1e-15), lambda_ub_input1_prop)
        mu_ub_input1_prop = torch.where(input1_unsettled, -(ub_input1_layer * lb_input1_layer) / (ub_input1_layer - lb_input1_layer + 1e-15), mu_ub_input1_prop)        

        # input2 is active
        lambda_lb_input2_prop = torch.where(input2_active, torch.ones(lb_input2_layer.size(), device=self.device), lambda_lb_input2_prop)
        lambda_ub_input2_prop = torch.where(input2_active, torch.ones(lb_input2_layer.size(), device=self.device), lambda_ub_input2_prop)
        # input2 is unsettled
        temp = torch.where(ub_input2_layer < -lb_input2_layer, torch.zeros(lb_input2_layer.size(), device=self.device), torch.ones(lb_input2_layer.size(), device=self.device))
        lambda_lb_input2_prop = torch.where(input2_unsettled, temp, lambda_lb_input2_prop)
        lambda_ub_input2_prop = torch.where(input2_unsettled, ub_input2_layer/(ub_input2_layer - lb_input2_layer + 1e-15), lambda_ub_input2_prop)
        mu_ub_input2_prop = torch.where(input2_unsettled, -(ub_input2_layer * lb_input2_layer) / (ub_input2_layer - lb_input2_layer + 1e-15), mu_ub_input2_prop)        

        # Checked 


        mu_lb = torch.zeros(lb_input1_layer.size(), device=self.device)
        mu_ub = torch.zeros(lb_input1_layer.size(), device=self.device)
        # case 1 x.ub <= 0 and y.ub <= 0
        # case 2 x.lb >=  0 and y.lb >= 0
        lambda_lb = torch.where(input1_active & input2_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(input1_active & input2_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub)
        # case 3 x.lb >= 0 and y.ub <= 0
        # lambda_lb = torch.where(input1_active & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        # lambda_ub = torch.where(input1_active & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        # mu_lb = torch.where(input1_active & input2_passive, lb_input1_layer, mu_lb)
        # mu_ub = torch.where(input1_active & input2_passive, ub_input1_layer, mu_ub)
        lambda_lb_input1 = torch.where(input1_active & input2_passive, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input1)
        lambda_ub_input1 = torch.where(input1_active & input2_passive, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input1)



        #case 4 (x.lb < 0 and x.ub > 0) and y.ub <= 0
        # lambda_lb = torch.where(input1_unsettled & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        # lambda_ub = torch.where(input1_unsettled & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        # mu_lb = torch.where(input1_unsettled & input2_passive, torch.zeros(lb_input1_layer.size(), device=self.device), mu_lb)
        # mu_ub = torch.where(input1_unsettled & input2_passive, ub_input1_layer, mu_ub)
        lambda_lb_input1 = torch.where(input1_unsettled & input2_passive, lambda_lb_input1_prop, lambda_lb_input1)
        lambda_ub_input1 = torch.where(input1_unsettled & input2_passive, lambda_ub_input1_prop, lambda_ub_input1)
        mu_ub = torch.where(input1_unsettled & input2_passive, mu_ub_input1_prop, mu_ub)

        #case 5 (x.ub <= 0) and y.lb >= 0
        # lambda_lb = torch.where(input1_passive & input2_active, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        # lambda_ub = torch.where(input1_passive & input2_active, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        # mu_lb = torch.where(input1_passive & input2_active, -ub_input2_layer, mu_lb)
        # mu_ub = torch.where(input1_passive & input2_active, -lb_input2_layer, mu_ub)
        lambda_lb_input2 = torch.where(input1_passive & input2_active, -torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input2)
        lambda_ub_input2 = torch.where(input1_passive & input2_active, -torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input2)

        # case 6 (x.ub <= 0) and (y.lb < 0 and y.ub > 0)
        # lambda_lb = torch.where(input1_passive & input2_unsettled, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        # lambda_ub = torch.where(input1_passive & input2_unsettled, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        # mu_lb = torch.where(input1_passive & input2_unsettled, -ub_input2_layer, mu_lb)
        # mu_ub = torch.where(input1_passive & input2_unsettled, torch.zeros(lb_input1_layer.size(), device=self.device), mu_ub)     
        lambda_lb_input2 = torch.where(input1_passive & input2_unsettled, -lambda_ub_input2_prop, lambda_lb_input2)
        lambda_ub_input2 = torch.where(input1_passive & input2_unsettled, -lambda_lb_input2_prop, lambda_ub_input2)
        mu_lb = torch.where(input1_passive & input2_unsettled, -mu_ub_input2_prop, mu_lb)


        # case 7 (x.lb >= 0) and (y.lb < 0 and y.ub > 0)
        lambda_lb = torch.where(input1_active & input2_unsettled, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(input1_active & input2_unsettled, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub)
        mu_lb = torch.where(input1_active & input2_unsettled, torch.min(lb_input1_layer, delta_lb_layer), mu_lb)
        mu_ub = torch.where(input1_active & input2_unsettled, torch.zeros(lb_input1_layer.size(), device=self.device), mu_ub)
        # lambda_lb_input1 = torch.where(input1_active & input2_unsettled, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input1)
        # lambda_ub_input1 = torch.where(input1_active & input2_unsettled, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input1)
        # lambda_lb_input2 = torch.where(input1_active & input2_unsettled, -lambda_ub_input2_prop, lambda_lb_input2)
        # lambda_ub_input2 = torch.where(input1_active & input2_unsettled, -lambda_lb_input2_prop, lambda_ub_input2)
        # mu_lb = torch.where(input1_active & input2_unsettled, -mu_ub_input2_prop, mu_lb)

        # case 8 (x.lb < 0 and x.ub > 0) and (y.lb >= 0)
        lambda_lb = torch.where(input1_unsettled & input2_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(input1_unsettled & input2_active, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_active, torch.zeros(lb_input1_layer.size(), device=self.device), mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_active, torch.max(delta_ub_layer, -lb_input2_layer), mu_ub)
        # lambda_lb_input1 = torch.where(input1_unsettled & input2_active, lambda_lb_input1_prop, lambda_lb_input1)
        # lambda_ub_input1 = torch.where(input1_unsettled & input2_active, lambda_ub_input1_prop, lambda_ub_input1)
        # mu_ub = torch.where(input1_unsettled & input2_active, mu_ub_input1_prop, mu_ub)
        # lambda_lb_input2 = torch.where(input1_unsettled & input2_active, -torch.ones(lb_input1_layer.size(), device=self.device), lambda_lb_input2)
        # lambda_ub_input2 = torch.where(input1_unsettled & input2_active, -torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub_input2)


        # case 9 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb >= 0)
        lambda_lb = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_lb)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size(), device=self.device), mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.zeros(lb_input1_layer.size(), device=self.device), mu_ub)
        # lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_active, torch.ones(lb_input1_layer.size(), device=self.device), lambda_ub)
        # if lambda.ub <= x.ub then delta_ub = delta
        # else delta_ub = x.ub
        case_9 = (input1_unsettled & input2_unsettled & delta_active)
        temp_lambda = torch.where((ub_input1_layer < delta_ub_layer) & case_9, lambda_ub_input1_prop, torch.ones(lb_input1_layer.size(), device=self.device))
        lambda_ub = torch.where(case_9 & (ub_input1_layer >= delta_ub_layer), temp_lambda, lambda_ub)
        lambda_ub_input1 = torch.where(case_9 & (ub_input1_layer < delta_ub_layer), temp_lambda, lambda_ub_input1)
        mu_ub = torch.where(case_9 & (ub_input1_layer < delta_ub_layer), mu_ub_input1_prop, mu_ub)
        
        # Checked        
        # case 10 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_ub <= 0)
        lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size(), device=self.device), lambda_ub)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size(), device=self.device), mu_lb)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_passive, torch.zeros(lb_input1_layer.size(), device=self.device), mu_ub)
        case_10 = (input1_unsettled & input2_unsettled & delta_passive)
        temp_lambda = torch.where((ub_input2_layer < delta_ub_layer) & case_10, -lambda_ub_input2_prop, torch.ones(lb_input1_layer.size(), device=self.device))
        lambda_lb = torch.where((ub_input2_layer >= delta_ub_layer) & case_10, temp_lambda, lambda_lb)
        lambda_lb_input2 = torch.where((ub_input2_layer < delta_ub_layer) & case_10, temp_lambda, lambda_lb_input2)
        mu_lb = torch.where((ub_input2_layer < delta_ub_layer) & case_10, -mu_ub_input2_prop, mu_lb)

        # case 11 (x.lb < 0 and x.ub > 0) and (y.lb < 0 and y.ub > 0) and (delta_lb < 0 and delta_ub > 0)
        # temp_mu = (delta_lb_layer * delta_ub_layer) / (delta_ub_layer - delta_lb_layer + 1e-15)
        # temp_lambda_lb = (-delta_lb_layer) / (delta_ub_layer - delta_lb_layer + 1e-15)
        # temp_lambda_ub = delta_ub_layer / (delta_ub_layer - delta_lb_layer + 1e-15)
        # lambda_lb = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, temp_lambda_lb, lambda_lb)
        # lambda_ub = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, temp_lambda_ub, lambda_ub)
        # mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, temp_mu, mu_lb)
        # mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, -temp_mu, mu_ub)
        lambda_lb_input1 = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, lambda_lb_input1_prop, lambda_lb_input1)
        lambda_ub_input1 = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, lambda_ub_input1_prop, lambda_ub_input1)
        mu_ub = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, mu_ub_input1_prop, mu_ub)
        lambda_lb_input2 = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, -lambda_ub_input2_prop, lambda_lb_input2)
        lambda_ub_input2 = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, -lambda_lb_input2_prop, lambda_ub_input2)
        mu_lb = torch.where(input1_unsettled & input2_unsettled & delta_unsettled, -mu_ub_input2_prop, mu_lb)

        # checked 
        # Segregate the +ve and -ve components of the coefficients
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_coef)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_coef)
        neg_comp_lb_input1, pos_comp_lb_input1 = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_input1_coef)
        neg_comp_ub_input1, pos_comp_ub_input1 = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_input1_coef)
        neg_comp_lb_input2, pos_comp_lb_input2 = self.pos_neg_weight_decomposition(back_prop_struct.delta_lb_input2_coef)
        neg_comp_ub_input2, pos_comp_ub_input2 = self.pos_neg_weight_decomposition(back_prop_struct.delta_ub_input2_coef)

        # self.log_file.write(f"lower bound input1 {lb_input1_layer}\n\n")
        # self.log_file.write(f"upper bound input1 {ub_input1_layer}\n\n")
        # self.log_file.write(f"lower bound input2 {lb_input2_layer}\n\n")
        # self.log_file.write(f"upper bound input2 {ub_input2_layer}\n\n")
        # self.log_file.write(f"lambda lb {lambda_lb}\n\n")
        # self.log_file.write(f"lambda ub {lambda_ub}\n\n")
        # self.log_file.write(f"lambda lb input1 {lambda_lb_input1}\n\n")
        # self.log_file.write(f"lambda ub input1 {lambda_ub_input1}\n\n")
        # self.log_file.write(f"lambda lb input2 {lambda_lb_input2}\n\n")
        # self.log_file.write(f"lambda ub input2 {lambda_ub_input2}\n\n")                        
        # self.log_file.write(f"mu lb {mu_lb}\n\n")
        # self.log_file.write(f"mu ub {mu_ub}\n\n")


        delta_lb_coef = pos_comp_lb * lambda_lb + neg_comp_lb * lambda_ub
        delta_lb_input1_coef = pos_comp_lb_input1 * lambda_lb_input1_prop + neg_comp_lb_input1 * lambda_ub_input1_prop
        delta_lb_input1_coef = delta_lb_input1_coef + pos_comp_lb * lambda_lb_input1 + neg_comp_lb * lambda_ub_input1
        delta_lb_input2_coef = pos_comp_lb_input2 * lambda_lb_input2_prop + neg_comp_lb_input2 * lambda_ub_input2_prop
        delta_lb_input2_coef = delta_lb_input2_coef + pos_comp_lb * lambda_lb_input2 + neg_comp_lb * lambda_ub_input2

     
        delta_ub_coef = pos_comp_ub * lambda_ub + neg_comp_ub * lambda_lb
        delta_ub_input1_coef = pos_comp_ub_input1 * lambda_ub_input1_prop + neg_comp_ub_input1 * lambda_lb_input1_prop
        delta_ub_input1_coef = delta_ub_input1_coef + pos_comp_ub * lambda_ub_input1 + neg_comp_ub * lambda_lb_input1
        delta_ub_input2_coef = pos_comp_ub_input2 * lambda_ub_input2_prop + neg_comp_ub_input2 * lambda_lb_input2_prop
        delta_ub_input2_coef = delta_ub_input2_coef + pos_comp_ub * lambda_ub_input2 + neg_comp_ub * lambda_lb_input2


        delta_lb_bias = pos_comp_lb @ mu_lb + neg_comp_lb @ mu_ub + back_prop_struct.delta_lb_bias
        delta_lb_bias = delta_lb_bias + neg_comp_lb_input1 @ mu_ub_input1_prop + neg_comp_lb_input2 @ mu_ub_input2_prop
        delta_ub_bias = pos_comp_ub @ mu_ub + neg_comp_ub @ mu_lb + back_prop_struct.delta_ub_bias
        delta_ub_bias = delta_ub_bias + pos_comp_ub_input1 @ mu_ub_input1_prop + pos_comp_ub_input2 @ mu_ub_input2_prop

        back_prop_struct.populate(delta_lb_coef=delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                                    delta_ub_coef=delta_ub_coef, delta_ub_bias=delta_ub_bias,
                                    delta_lb_input1_coef=delta_lb_input1_coef, 
                                    delta_ub_input1_coef=delta_ub_input1_coef,
                                    delta_lb_input2_coef=delta_lb_input2_coef,
                                    delta_ub_input2_coef=delta_ub_input2_coef)

        return back_prop_struct


    def get_layer_size(self, linear_layer_index):
        layer = self.net[self.linear_conv_layer_indices[linear_layer_index]]
        if layer.type is LayerType.Linear:
            shape = self.shapes[linear_layer_index + 1]
            return shape
        if layer.type is LayerType.Conv2D:
            shape = self.shapes[linear_layer_index+ 1]
            return (shape[0] * shape[1] * shape[2])
        

    # layer index : index of the current layer
    # linear layer index: No of linear layers seen before the current layer.
    def back_substitution(self, layer_index, linear_layer_index, delta_lbs, delta_ubs):
        if linear_layer_index != len(delta_lbs):
            raise ValueError("Size of lower bounds computed in previous layers don't match")

        back_prop_struct = None
        delta_lb = None
        delta_ub = None

        self.log_file.write(f"------ Starting for {layer_index} -----------\n\n")
        for i in reversed(range(layer_index + 1)):
            # Concretize the bounds for the previous layers.
            self.log_file.write(f"******* Back propagation for layer index {layer_index} current layer index {i} *******\n\n")
            if self.net[i].type in [LayerType.Linear, LayerType.Conv2D] and back_prop_struct is not None:
                new_delta_lb, new_delta_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,delta_lb_layer=delta_lbs[linear_layer_index], 
                                                                    delta_ub_layer=delta_ubs[linear_layer_index],
                                                         lb_input1_layer=self.lb_input1[linear_layer_index], 
                                                ub_input1_layer=self.ub_input1[linear_layer_index], 
                                                lb_input2_layer=self.lb_input2[linear_layer_index], 
                                                ub_input2_layer=self.ub_input2[linear_layer_index])
                if not torch.all(new_delta_lb <= new_delta_ub + 1e-6) :
                    print(f"Issue {new_delta_lb  - new_delta_ub }\n\n")
                    self.log_file.write(f"Issue {new_delta_lb  - new_delta_ub }\n\n")
                    if delta_lb is not None:
                        self.log_file.write(f"curr delta diff {delta_lb - delta_ub}\n\n")
                    assert torch.all(new_delta_lb <= new_delta_ub + 1e-6)
                
                delta_lb = (new_delta_lb if delta_lb is None else (torch.max(delta_lb, new_delta_lb)))
                delta_ub = (new_delta_ub if delta_ub is None else (torch.min(delta_ub, new_delta_ub)))

            if back_prop_struct is None:
                layer_size = self.get_layer_size(linear_layer_index=linear_layer_index)
                delta_lb_coef = torch.eye(n=layer_size, device=self.device)
                delta_lb_bias = torch.zeros(layer_size, device=self.device)
                delta_ub_coef = torch.eye(n=layer_size, device=self.device)
                delta_ub_bias = torch.zeros(layer_size, device=self.device)
                delta_lb_input1_coef = torch.zeros((layer_size, layer_size), device=self.device)                
                delta_ub_input1_coef = torch.zeros((layer_size, layer_size), device=self.device)
                delta_lb_input2_coef = torch.zeros((layer_size, layer_size), device=self.device)
                delta_ub_input2_coef = torch.zeros((layer_size, layer_size), device=self.device)
                back_prop_struct = DiffPropStruct()
                back_prop_struct.populate(delta_lb_coef=delta_lb_coef, delta_lb_bias=delta_lb_bias, 
                                          delta_ub_coef=delta_ub_coef, delta_ub_bias=delta_ub_bias,
                                          delta_lb_input1_coef=delta_lb_input1_coef, 
                                          delta_ub_input1_coef=delta_ub_input1_coef,
                                          delta_lb_input2_coef=delta_lb_input2_coef,
                                          delta_ub_input2_coef=delta_ub_input2_coef)

            curr_layer = self.net[i] 
            if curr_layer.type is LayerType.Linear:
               back_prop_struct = self.handle_linear(linear_wt=curr_layer.weight,
                                     bias=curr_layer.bias, back_prop_struct=back_prop_struct)
               linear_layer_index -= 1
            elif curr_layer.type is LayerType.Conv2D:
                back_prop_struct = self.handle_conv(conv_weight=curr_layer.weight, conv_bias=curr_layer.bias, 
                                        back_prop_struct=back_prop_struct, 
                                        preconv_shape=self.shapes[linear_layer_index], postconv_shape=self.shapes[linear_layer_index + 1],
                                        stride=curr_layer.stride, padding=curr_layer.padding, dilation=curr_layer.dilation)
                linear_layer_index -= 1
            elif curr_layer.type is LayerType.ReLU:
                back_prop_struct = self.handle_relu(
                                                back_prop_struct=back_prop_struct,
                                                lb_input1_layer=self.lb_input1[linear_layer_index], 
                                                ub_input1_layer=self.ub_input1[linear_layer_index], 
                                                lb_input2_layer=self.lb_input2[linear_layer_index], 
                                                ub_input2_layer=self.ub_input2[linear_layer_index],
                                                delta_lb_layer=delta_lbs[linear_layer_index], 
                                                delta_ub_layer=delta_ubs[linear_layer_index])
            else:
                raise NotImplementedError(f'diff verifier for {curr_layer.type} is not implemented')
        
        # Compute the bounds after back substituting the bounds to the input layer.         
        new_delta_lb, new_delta_ub = self.concretize_bounds(back_prop_struct=back_prop_struct,
                                                            delta_lb_layer=self.diff, 
                                                            delta_ub_layer=self.diff,
                                                            lb_input1_layer=self.lb_input1[-1],
                                                            ub_input1_layer=self.ub_input1[-1],
                                                            lb_input2_layer=self.lb_input2[-1],
                                                            ub_input2_layer=self.ub_input2[-1])
                    
        if not torch.all(new_delta_lb <= new_delta_ub + 1e-6) :
            print(f"Issue {new_delta_lb  - new_delta_ub }\n\n")
            self.log_file.write(f"Issue {new_delta_lb  - new_delta_ub }\n\n")
            if delta_lb is not None:
                self.log_file.write(f"curr delta diff {delta_lb - delta_ub}\n\n")
            assert torch.all(new_delta_lb <= new_delta_ub + 1e-6)
        delta_lb = (new_delta_lb if delta_lb is None else (torch.max(delta_lb, new_delta_lb)))
        delta_ub = (new_delta_ub if delta_ub is None else (torch.min(delta_ub, new_delta_ub)))


        return delta_lb, delta_ub
    
    def swap_inputs(self):
        self.lb_input1, self.lb_input2 = (self.lb_input2, self.lb_input1)
        self.ub_input1, self.ub_input2 = (self.ub_input2, self.ub_input1)
        self.diff = -self.diff        

    def run(self):
        self.log_file = open(self.diff_log_filename, 'w+')
        delta_lbs = []
        delta_ubs = []
        if self.net is None:
            raise ValueError("Passed network can not be none")
        for ind, layer in enumerate(self.net):
            if layer.type in [LayerType.Linear, LayerType.Conv2D]:
                self.linear_conv_layer_indices.append(ind)
        
                
        if len(self.lb_input1) - 1 != len(self.linear_conv_layer_indices) or len(self.ub_input1) - 1 != len(self.linear_conv_layer_indices):
            raise ValueError("Input1 bounds do not match")
        if len(self.lb_input2) - 1 != len(self.linear_conv_layer_indices) or len(self.ub_input2) - 1 != len(self.linear_conv_layer_indices):
            raise ValueError("Input2 bounds do not match")

        for linear_layer_index, layer_index in enumerate(self.linear_conv_layer_indices):
            curr_delta_lb, curr_delta_ub = self.back_substitution(layer_index=layer_index, linear_layer_index=linear_layer_index,
                                                                   delta_lbs=delta_lbs, delta_ubs=delta_ubs)
            
            if not torch.all(curr_delta_lb <= curr_delta_ub + 1e-6) :
                print(f"Issue {curr_delta_lb  - curr_delta_ub }\n\n")
                assert torch.all(curr_delta_lb <= curr_delta_ub + 1e-6)
            brute_delta_lb = self.lb_input1[linear_layer_index] - self.ub_input2[linear_layer_index]
            brute_delta_ub = self.ub_input1[linear_layer_index] - self.lb_input2[linear_layer_index]            
            self.log_file.write(f"curr_delta_lb {linear_layer_index} {curr_delta_lb}\n\n")            
            self.log_file.write(f'curr_diff_delta_lb {linear_layer_index} {brute_delta_lb}\n\n')
            self.log_file.write(f"curr_delta_ub {linear_layer_index} {curr_delta_ub}\n\n")            
            self.log_file.write(f'curr_diff_delta_ub {linear_layer_index} {brute_delta_ub}\n\n')
            curr_delta_lb = torch.max(brute_delta_lb, curr_delta_lb)
            curr_delta_ub = torch.min(brute_delta_ub, curr_delta_ub)
            delta_lbs.append(curr_delta_lb)
            delta_ubs.append(curr_delta_ub)

        # delta_lbs_reverse = []
        # delta_ubs_reverse = []        
        # # Reverse computation (y - x)
        # # First swap input1 and input2
        # self.swap_inputs()
        # self.log_file.write("Started in reverse direction \n\n")
        # for linear_layer_index, layer_index in enumerate(self.linear_conv_layer_indices):
        #     curr_delta_lb, curr_delta_ub = self.back_substitution(layer_index=layer_index, 
        #                                                           linear_layer_index=linear_layer_index,
        #                                                           delta_lbs=delta_lbs_reverse, delta_ubs=delta_ubs_reverse)
            
        #     if not torch.all(curr_delta_lb <= curr_delta_ub + 1e-6) :
        #         print(f"Issue {curr_delta_lb  - curr_delta_ub }\n\n")
        #         assert torch.all(curr_delta_lb <= curr_delta_ub + 1e-6)
        #     brute_delta_lb = self.lb_input1[linear_layer_index] - self.ub_input2[linear_layer_index]
        #     brute_delta_ub = self.ub_input1[linear_layer_index] - self.lb_input2[linear_layer_index]            
        #     self.log_file.write(f"curr_delta_lb {linear_layer_index} {curr_delta_lb}\n\n")            
        #     self.log_file.write(f'curr_diff_delta_lb {linear_layer_index} {brute_delta_lb}\n\n')
        #     self.log_file.write(f"curr_delta_ub {linear_layer_index} {curr_delta_ub}\n\n")            
        #     self.log_file.write(f'curr_diff_delta_ub {linear_layer_index} {brute_delta_ub}\n\n')
        #     curr_delta_lb = torch.max(brute_delta_lb, curr_delta_lb)
        #     curr_delta_ub = torch.min(brute_delta_ub, curr_delta_ub)
        #     delta_lbs_reverse.append(curr_delta_lb)
        #     delta_ubs_reverse.append(curr_delta_ub)
        
        # # Compute final lbs, ubs
        # final_delta_lbs = []
        # final_delta_ubs = []

        # for i, delta_lb in enumerate(delta_lbs):
        #     final_delta_lb = torch.maximum(delta_lb, -delta_ubs_reverse[i])
        #     final_delta_ub = torch.minimum(delta_ubs[i], -delta_lbs_reverse[i])
        #     final_delta_lbs.append(final_delta_lb)
        #     final_delta_ubs.append(final_delta_ub)
        #     self.log_file.write(f"final_delta_lb {i} {final_delta_lb}\n\n")          
        #     self.log_file.write(f"final_delta_ub {i} {final_delta_ub}\n\n")           
        self.log_file.close()

        return delta_lbs, delta_ubs