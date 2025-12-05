import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def get_cifar_loaders(dataset_name='cifar10', data_root='./data', batch_size=128, num_workers=4):
    """
    Returns CIFAR-10 or CIFAR-100 train and test loaders.
    """
    # Ensure data directory exists
    os.makedirs(data_root, exist_ok=True)

    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        Dataset = torchvision.datasets.CIFAR10
    elif dataset_name == 'cifar100':
        if data_root == './data': # Default override
            data_root = '/data/cifar100'
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        Dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = Dataset(root=data_root, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = Dataset(root=data_root, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
