import torch
from unittest import TestCase
from src.domains.diff_deeppoly import DiffDeepPoly
from torch.nn import Conv2d
import numpy as np
from src.common.network import Layer, LayerType

class DiffDeepPolyTest(TestCase):

    # Unittest for the linear layer.
    def initialize_verifier(self):
        input1 = torch.rand(2)
        input2 = torch.rand(2)
        net = []
        lb_input1 = input1 - 0.1
        ub_input1 = input1 + 0.1  
        lb_input2 = input2  - 0.2 
        ub_input2 = input2 + 0.2
        diff_deep_poly_ver = DiffDeepPoly(input1, input2, net, lb_input1, ub_input1, lb_input2, ub_input2, device='cpu')
        return diff_deep_poly_ver
    
    def test_diff_deeppoly_linear(self):
        diff_deep_poly_ver = self.initialize_verifier()
        weight_mat = torch.rand(2, 3)
        bias = None
        lb_coef = torch.rand(3, 2)
        lb_bias = torch.rand(3)
        ub_coef = torch.rand(3, 2)
        ub_bias = torch.rand(3)
        delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias = diff_deep_poly_ver.handle_linear(weight_mat, 
                                                                                                      bias, lb_coef, 
                                                                                                      lb_bias, ub_coef, 
                                                                                                      ub_bias)
        
        assert(torch.all(delta_lb_coef == (lb_coef @ weight_mat)))
        assert(torch.all(delta_ub_coef == (ub_coef @ weight_mat)))
        assert(torch.all(delta_lb_bias == lb_bias))
        assert(torch.all(delta_ub_bias == ub_bias))

    # Unittest for the conv layer.
    def test_diff_deeppoly_conv(self):
        diff_deep_poly_ver = self.initialize_verifier()
        in_channels = 1
        out_channels = 2
        kernel_size = (2, 2)
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)
        groups = 1
        with torch.no_grad():
            conv_layer = Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=groups, 
                                    bias=True,
                                    device='cpu')
            input = torch.rand(1 , 3, 3)
            preconv_shape = (1, 3, 3)
            postconv_shape = (2, 2, 2)
            lb_coef = torch.rand(3, 8)
            lb_bias = torch.rand(3)
            ub_coef = torch.rand(3, 8)
            ub_bias = torch.rand(3)
            conv_layer_weight = conv_layer.weight
            conv_layer_bias = conv_layer.bias
            print("Conv layer bias shape ", conv_layer_bias.shape)
            delta_lb_coef, delta_lb_bias, delta_ub_coef, delta_ub_bias = diff_deep_poly_ver.handle_conv(conv_layer_weight, conv_layer_bias, 
                                                                                        lb_coef, lb_bias, 
                                ub_coef, ub_bias, preconv_shape, postconv_shape,
                                stride, padding, groups, dilation)
            output = conv_layer(input)
            output = output.reshape(-1)
            input_flat = input.reshape(-1)
            output_lb = lb_coef @ output
            output_ub = ub_coef @ output
            output_lb = output_lb.numpy()
            output_ub = output_ub.numpy()
            final_lb_bias = lb_bias + torch.sum(lb_coef[:, :4], dim=1) * conv_layer_bias[0].item() + torch.sum(lb_coef[:, 4:], dim=1) * conv_layer_bias[1].item()
            final_ub_bias = ub_bias + torch.sum(ub_coef[:, :4], dim=1) * conv_layer_bias[0].item() + torch.sum(ub_coef[:, 4:], dim=1) * conv_layer_bias[1].item()            
            print(final_lb_bias.shape)
            output_lb1 = (delta_lb_coef @ input_flat) + final_lb_bias - lb_bias
            output_ub1 = (delta_ub_coef @ input_flat) + final_ub_bias - ub_bias
            output_lb1 = output_lb1.numpy()
            output_ub1 = output_ub1.numpy()        
            assert(np.allclose(output_lb, output_lb1))
            assert(np.allclose(output_ub, output_ub1))
            assert(np.allclose(final_lb_bias.numpy(), delta_lb_bias.numpy()))
            assert(np.allclose(final_ub_bias.numpy(), delta_ub_bias.numpy()))

    def test_full_propagation(self):
        network = []
        weight1 = torch.tensor([[2, -1], [3, -1]], dtype=torch.float)
        weight2 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float)
        network.append(Layer(weight=weight1, bias=torch.zeros(2), type=LayerType.Linear))
        network.append(Layer(type=LayerType.ReLU))
        network.append(Layer(weight=weight2, bias=torch.zeros(2), type=LayerType.Linear))
        input1 = torch.tensor([1, 1], dtype=torch.float)
        input2 = torch.tensor([2, 2], dtype=torch.float)
        lb_input1 = [torch.tensor([-2, -2], dtype=torch.float), torch.tensor([-6, -4], dtype=torch.float)]
        ub_input1 = [torch.tensor([4, 6], dtype=torch.float), torch.tensor([4, -6], dtype=torch.float)]
        lb_input2 = [torch.tensor([-1, 0], dtype=torch.float), torch.tensor([-8, -5], dtype=torch.float)]
        ub_input2 = [torch.tensor([5, 8], dtype=torch.float), torch.tensor([5, 8], dtype=torch.float)]
        diff_deep_poly_ver = DiffDeepPoly(input1, input2, network, lb_input1, ub_input1, lb_input2, ub_input2, device='cpu')
        delta_lbs, delta_ubs = diff_deep_poly_ver.run()
        assert(np.allclose(delta_lbs[0].numpy(), torch.tensor([-1, -2],  dtype=torch.float).numpy()))
        assert(np.allclose(delta_ubs[0].numpy(), torch.tensor([-1, -2],  dtype=torch.float).numpy()))
        assert(np.allclose(delta_lbs[1].numpy(), torch.tensor([-1, -2],  dtype=torch.float).numpy()))
        assert(np.allclose(delta_ubs[1].numpy(), torch.tensor([2, 1],  dtype=torch.float).numpy()))         