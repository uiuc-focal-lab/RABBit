'''
Copied from https://github.com/uiuc-arc/FANC/blob/main/proof_transfer/approximate.py
WIP to work with this repo
Generate the approximated networks provided the original network. The input is in saved pytorch format.
The generated outputs are in ONNX format.
'''
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn

from torch.nn import functional as F
from enum import Enum
from src import util
from src.common.network import LayerType
from src.common.dataset import Dataset

DEVICE = 'cpu'


class QuantizationType(Enum):
    INT8 = 1
    INT16 = 2
    INT32 = 3
    FP16 = 4


class Quantize:
    def __init__(self, qt_type):
        self.qt_type = qt_type

    def __repr__(self):
        return str(self.qt_type)

    def approximate(self, net_name, dataset):
        net = util.get_net(net_name, dataset)
        check_accuracy(net, dataset)

        if self.qt_type == QuantizationType.INT8:
            dummy_quant(net, 8)
        elif self.qt_type == QuantizationType.INT16:
            dummy_quant(net, 16)
        elif self.qt_type == QuantizationType.FP16:
            dummy_quant_float(net)
        else:
            raise ValueError("Unsupported approximation!")
        return net


class Prune:
    def __init__(self, percent):
        self.percent = percent

    def __repr__(self):
        return 'prune'+str(self.percent)

    def approximate(self, net_name, dataset):
        net = util.get_net(net_name, dataset)
        check_accuracy(net, dataset)
        prune_model(net, dataset, prune_percent=self.percent)
        return net


class Random:
    def __init__(self, ptb_eps, layers=None):
        self.ptb_eps = ptb_eps

    def __repr__(self):
        return 'random'+str(self.ptb_eps)

    def approximate(self, net_name, dataset):
        net = util.get_net(net_name, dataset)
        check_accuracy(net, dataset)
        for layer in net:
            if layer.type is not LayerType.Linear:
                continue

            rdp = -self.ptb_eps + 2*torch.rand(layer.weight.shape)*self.ptb_eps
            layer.weight = layer.weight + rdp
        return net


def get_approx_net_name(net_name, approx_type):
    tmp_str = net_name.split('.')
    tmp_str[-1:-1] = [str(approx_type).split('.')[-1]]
    return ".".join(tmp_str)


def prune_model(model, dataset, skip_layer=0, prune_percent=50, post_finetune=False):
    density(model)
    prev_accuracy = check_accuracy(model, dataset)

    prune_weights(model, prune_percent, skip_layer=skip_layer)

    check_accuracy(model, dataset)
    density(model)

    print("Fine tune the network to get accuracy:", prev_accuracy)

    if post_finetune:
        finetune(model, dataset, req_accuracy=prev_accuracy)

    density(model)


def prepare_data(dataset, train):
    print('==> Preparing data..')

    if dataset == Dataset.CIFAR10:
        transform_test = transforms.Compose([
            transforms.ToTensor()
            ,
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        )

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=2)

        inputs, _ = next(iter(testloader))
    elif dataset == Dataset.MNIST:
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=train, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0,), (1,))
                                       ])),
            batch_size=1, shuffle=True)
    else:
        raise ValueError("Unsupported dataset!")
    return testloader


def prune_weights(net, per, skip_layer=0):
    for layer in net:
        if layer.type is not LayerType.Linear:
            continue

        weight = layer.weight

        per_it = per
        if skip_layer > 0:
            skip_layer -= 1
            per_it = 0

        print('Pruning layer: ', layer.type, ' | Percentage: ', per_it)
        cutoff = np.percentile(np.abs(weight), per_it)

        if len(weight.shape) == 2:
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    if abs(weight[i][j]) < cutoff:
                        weight[i][j] = 0

        elif len(weight.shape) == 1:
            for i in range(weight.shape[0]):
                if abs(weight[i]) < cutoff:
                    weight[i] = 0


def density(net):
    count = 0
    count_nz = 0

    for layer in net:
        if layer.type is not LayerType.Linear:
            continue

        weight = layer.weight

        # Transform the parameter as required.
        if len(weight.shape) == 2:
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    count += 1
                    if weight[i][j] != 0:
                        count_nz += 1

        elif len(weight.shape) == 1:
            for i in range(weight.shape[0]):
                count += 1
                if weight[i][j] != 0:
                    count_nz += 1

    print('Density :', count_nz * 1.0 / count)


'''
WIP: Get bounds for a layer using interval propogation. 
TODO: 
1. Use multiple layers
2. Refactor the propagation code to a separate analyzer module
'''


def get_bounds(images, params):
    eps = 0.05
    is_conv = True
    if not is_conv:

        lb = (images - eps).reshape(images.shape[0], -1)
        ub = (images + eps).reshape(images.shape[0], -1)

        pos_wt = F.relu(params[0])
        neg_wt = -F.relu(-params[0])

        oub = F.relu(ub @ pos_wt.T + lb @ neg_wt.T)
        olb = F.relu(lb @ pos_wt.T + ub @ neg_wt.T)
    else:
        lb = (images - eps).reshape(images.shape[0], -1)
        ub = (images + eps).reshape(images.shape[0], -1)

        weight = params[0]
        bias = params[1]

        num_kernel = weight.shape[0]

        k_h, k_w = 4, 4
        s_h, s_w = 2, 2
        p_h, p_w = 1, 1

        input_h, input_w = 28, 28

        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        linear_cof = []

        size = 784
        shape = (1, 28, 28)

        cof = torch.eye(size).reshape(size, *shape)
        pad2d = (p_w, p_w, p_h, p_h)
        cof = F.pad(cof, pad2d)

        for i in range(output_h):
            w_cof = []
            for j in range(output_w):
                h_start = i * s_h
                h_end = h_start + k_h
                w_start = j * s_w
                w_end = w_start + k_w

                w_cof.append(cof[:, :, h_start: h_end, w_start: w_end])

            linear_cof.append(torch.stack(w_cof, dim=1))

        linear_cof = torch.stack(linear_cof, dim=1).reshape(size, output_h, output_w, -1)

        new_weight = weight.reshape(num_kernel, -1).T
        new_cof = linear_cof @ new_weight
        new_cof = new_cof.permute(0, 3, 1, 2).reshape(size, -1)

        pos_wt = F.relu(new_cof)
        neg_wt = -F.relu(-new_cof)

        bias = bias.view(-1, 1, 1).expand(num_kernel, output_h, output_w).reshape(1, -1)

        oub = F.relu(ub @ pos_wt.T + lb @ neg_wt.T)
        olb = F.relu(lb @ pos_wt.T + ub @ neg_wt.T)

    return olb, oub


'''
WIP: Fine-tune the model to be more amenable to the proof transfer. 
'''


def finetune(model, dataset, epochs=5, req_accuracy=1.0):
    loss_type = 'sum'

    if dataset == 'mnist':
        if loss_type == 'volume':
            learning_rate = 1e-7
        elif loss_type == 'sum':
            learning_rate = 1e-7
    elif dataset == 'cifar10':
        learning_rate = 1e-7
    else:
        raise ValueError("Unsupported dataset!")

    num_epochs = epochs

    # Extract all the weights of the model
    params = []
    for param in model.parameters():
        params.append(param)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [params[0]],
        lr=learning_rate)

    # Train the model
    total_step = len(trainloader)
    first_time = True

    all_ims = []

    for epoch in range(num_epochs):
        cur_accuracy = check_accuracy(model, testloader)

        if cur_accuracy >= req_accuracy:
            break

        for i, (images, labels) in enumerate(trainloader):
            all_ims.append(images)
            # Move tensors to the configured device
            # images = images.reshape(-1, 28 * 28).to(DEVICE)
            if dataset == 'mnist':
                images = images.reshape(-1, 1, 28, 28).to(DEVICE)
            else:
                images = images.reshape(-1, 3, 32, 32).to(DEVICE)

            labels = labels.to(DEVICE)
            olb, oub = get_bounds(images, params)

            if loss_type == 'volume':
                loss2 = torch.sum(torch.log(oub - olb + 1))
                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()

            elif loss_type == 'sum':
                loss2 = torch.sum(oub - olb)
                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()

            elif loss_type == 'mix':
                # Forward pass
                outputs = model(images)
                print(olb.shape)
                loss2 = torch.sum(oub - olb)
                print(loss, olb[0], oub[0])
                loss1 = criterion(outputs, labels)

                if first_time:
                    gamma = (loss1 / loss2).detach()
                    first_time = False

                loss = loss1 + gamma * loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss2.item()))
                # print(loss2.item(), loss2.item())

        olb, oub = get_bounds(all_ims[0], params)
        loss2 = torch.sum(oub - olb)
        print(loss2.item())
        print((oub - olb)[:20])


"""
Currently only works with onnx
"""


def check_accuracy(net, dataset):
    if dataset == Dataset.ACAS:
        return None

    testloader = prepare_data(dataset, False)
    inputs, _ = next(iter(testloader))

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = util.compute_output_tensor(inputs, net)

            predicted = outputs[1].argmax()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Accuracy: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
    return correct / total


def dummy_quant(net, quant_bit):
    # Dummy quantized
    # Calculate max to do the quantization symmetric, per-tensor

    def quant(x, scale):
        return int(x * scale)

    def unquant(x, scale):
        return x / scale

    for layer in net:
        if layer.type is not LayerType.Linear:
            continue

        weight = layer.weight
        if len(weight.shape) == 2:
            abs_max = weight.abs().max()

            scale = (2 ** (quant_bit - 1)) / abs_max

            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    weight[i][j] = unquant(quant(weight[i][j], scale), scale)

        elif len(layer.weight.shape) == 1:
            abs_max = weight.abs().max()
            scale = (2 ** (quant_bit - 1)) / abs_max

            for i in range(weight.shape[0]):
                weight[i] = unquant(quant(weight[i], scale), scale)
        else:
            print('Param shape length is: ', len(weight.shape))

        # Update the weight
        layer.weight = weight


def dummy_quant_float(net):
    # Dummy quantized
    # Calculate max to do the quantization symmetric, per-tensor

    def quant(x):
        return torch.Tensor([float(np.float16(x.item()))])

    for layer in net:
        if layer.type is not LayerType.Linear:
            continue

        weight = layer.weight
        if len(weight.shape) == 2:
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    weight[i][j] = quant(weight[i][j])

        elif len(layer.weight.shape) == 1:
            for i in range(weight.shape[0]):
                weight[i] = quant(weight[i])
        else:
            print('Param shape length is: ', len(weight.shape))

        # Update the weight
        layer.weight = weight
