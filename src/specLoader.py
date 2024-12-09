import torch
import torchvision
import torchvision.transforms as transforms
from src.common import Dataset, RavenMode
from auto_LiRPA.utils import get_spec_matrix
from raven.src.network_conversion_helper import convert_model
from raven.src.config import mnist_data_transform

def get_transforms(dataset, transform=True):
    if dataset is Dataset.CIFAR10:
        return transforms.Compose([transforms.ToTensor()] + ([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] if transform else []))
    elif dataset is Dataset.MNIST:
        return transforms.Compose([ transforms.ToTensor()]
                                        + ([transforms.Normalize((0.1307,), (0.3081,))] if transform else []))
    else:
        raise ValueError(f'Dataset {dataset} not recognised')

# Get the standard deviation use to scale to epsilon.
def get_std(dataset, transform=True):
    if not transform:
        return 1.0
    if dataset is Dataset.CIFAR10:
        return torch.tensor([0.2023, 0.1994, 0.2010])
    elif dataset is Dataset.MNIST:
        return 0.3081
    else:
        raise ValueError(f'Dataset {dataset} not recognised')

def get_mean(dataset, transform = True):
    if not transform:
        return torch.tensor([0.0])
    if dataset is Dataset.CIFAR10:
        return torch.tensor([0.4914, 0.4822, 0.4465])
    elif dataset is Dataset.MNIST:
        return torch.tensor([0.1307])
    else:
        raise ValueError(f'Dataset {dataset} not recognised')

def get_data_min_max(dataset, transform = True):
    mean = get_mean(dataset, transform)
    
    std = get_std(dataset, transform)
    std = std if torch.is_tensor([std]) else torch.tensor(std)

    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
    return data_min, data_max
    
def prepare_data(dataset, train=False, batch_size=100, transform=False):
    if dataset == Dataset.CIFAR10:
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(), ])

        transform_test = get_transforms(dataset=dataset, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)

        inputs, _ = next(iter(testloader))
    elif dataset == Dataset.MNIST:
        transform_test = get_transforms(dataset=dataset, transform=transform)
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=train, download=True,
                                       transform=transform_test),
            batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Unsupported Dataset")
    return testloader


def filter_misclassified(nets, inputs, labels):
    if nets[0] is None:
        return inputs, labels
    try:
        with torch.no_grad():
            converted_model = convert_model(nets[0], remove_last_layer=False, all_linear=False)
            outputs = converted_model(inputs)
            output_labels = torch.max(outputs, axis=1)[1]
            # print(f'matching tensor {output_labels == labels}')
            print(f"accuracy {(output_labels == labels).sum() / labels.shape[0] * 100}")
            inputs = inputs[output_labels == labels]
            labels = labels[output_labels == labels]
            return inputs, labels
    except:
        try:
            outputs = nets[0](inputs)
            output_labels = torch.max(outputs, axis=1)[1]
            # print(f'matching tensor {output_labels == labels}')
            print(f"accuracy {(output_labels == labels).sum() / labels.shape[0] * 100}")
            inputs = inputs[output_labels == labels]
            labels = labels[output_labels == labels]
            return inputs, labels
        except:
            print('\n Warning: can not convert to pytorch model')
            return inputs, labels

def get_input_bounds(images, eps, dataset, transform):
    eps_tensor = eps / get_std(dataset=dataset, transform=transform)
    print(f'eps tensor {eps_tensor}')
    lbs, ubs = [], []
    for img in images:
        img_shape = img.shape
        lb = img.view(img_shape[0], -1).T - eps_tensor
        ub = img.view(img_shape[0], -1).T + eps_tensor
        lb = lb.T.view(img_shape)
        ub = ub.T.view(img_shape)
        lbs.append(lb)
        ubs.append(ub)
    lbs = torch.stack(lbs, dim=0)
    ubs = torch.stack(ubs, dim=0)
    return lbs, ubs

def process_input_for_binary(inputs, labels, target_count=0):
    new_inputs = []
    new_labels = []
    count = 0
    binary_label = [0, 1]
    for i in range(len(inputs)):
        if labels[i] in binary_label and count < target_count:
            new_inputs.append(inputs[i])
            new_labels.append(labels[i])
            count += 1
    new_inputs = torch.stack(new_inputs)
    new_labels = torch.stack(new_labels)
    return new_inputs, new_labels

def get_binary_spec(dataset : Dataset, raven_mode : RavenMode,
                     count, nets, eps, dataloading_seed, net_names):
    assert dataset == Dataset.MNIST
    transform = mnist_data_transform(dataset=dataset, net_name=net_names[0])
    testloader = prepare_data(dataset=dataset, train=False, batch_size=count*15, transform=transform)
    images, labels = next(iter(testloader))
    images, labels = filter_misclassified(nets=nets, inputs=images, labels=labels)
    images, labels = process_input_for_binary(inputs=images, labels=labels, target_count=count)
    if raven_mode in [RavenMode.UAP_BINARY]:
        constraint_matrices = get_spec_matrix(images, labels.long(), 10)
        # org_shape = constraint_matrices.shape
        # assert len(org_shape) == 3
        # length = constraint_matrices.shape[0]
        # idx1 = [i for i in range(length)]
        # idx2 = [0 for i in range(length)]
        # constraint_matrices = constraint_matrices[idx1, idx2].reshape((org_shape[0], 1, org_shape[2]))
        for x in constraint_matrices:
            for j in range(1, 9):
                x[j] = x[0]
        print(f'constraint matrices shape {constraint_matrices.shape}')
        print(f'constraint matrice for label {labels[0]}\n\n {constraint_matrices[0]}')
        print(f'constraint matrice for label {labels[1]}\n\n {constraint_matrices[1]}')
    else:
        raise ValueError(f'Output specification of {raven_mode} is not supported')
    lbs, ubs = get_input_bounds(images=images, eps=eps, dataset=dataset, transform=transform)
    return images, labels, constraint_matrices, lbs, ubs


def get_specification(dataset : Dataset, raven_mode : RavenMode,
                     count, nets, eps, dataloading_seed, net_names):
    assert len(net_names) > 0
    if raven_mode is RavenMode.UAP_BINARY:
        print(f'\nloading UAP Binary')
        return get_binary_spec(dataset=dataset, raven_mode=raven_mode, count=count, 
                               nets=nets, eps=eps, dataloading_seed=dataloading_seed, 
                               net_names=net_names)
    transform = mnist_data_transform(dataset=dataset, net_name=net_names[0])
    testloader = prepare_data(dataset=dataset, train=False, batch_size=count*3, transform=transform)
    images, labels = next(iter(testloader))
    images, labels = filter_misclassified(nets=nets, inputs=images, labels=labels)
    images, labels = images[:count], labels[:count]
    if raven_mode in [RavenMode.UAP, RavenMode.ENSEMBLE]:
        constraint_matrices = get_spec_matrix(images, labels.long(), 10)
    else:
        raise ValueError(f'Output specification of {raven_mode} is not supported')
    lbs, ubs = get_input_bounds(images=images, eps=eps, dataset=dataset, transform=transform)
    return images, labels, constraint_matrices, lbs, ubs
    