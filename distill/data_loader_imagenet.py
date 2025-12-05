import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def get_imagenet_loaders(data_root='/data/ImageNet', batch_size=256, num_workers=8):
    """
    Returns ImageNet train and val loaders.
    Assumes standard ImageNet structure:
        data_root/train/class_xxx/img.jpg
        data_root/val/class_xxx/img.jpg
    """
    
    # ImageNet standard transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_root, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_root, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
