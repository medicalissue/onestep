import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from distill.models import ResNetWrapper, SimpleCNN
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    # 2. Check Teacher Accuracy
    log.info("Checking Teacher (ResNet18) Accuracy from HF...")
    teacher = ResNetWrapper(num_classes=10, pretrained=True, repo_id="edadaltocg/resnet18_cifar10").to(device)
    teacher.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = teacher(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    log.info(f"Teacher Accuracy: {acc:.2f}%")
    
    if acc < 15:
        log.error("Teacher is broken! Accuracy is near random.")
        return

    # 3. Check Student Learning Capability (Standard CE)
    log.info("Checking Student (SimpleCNN) Learning Capability (Standard CE)...")
    student = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    student.train()
    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = student(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            _, predicted = outputs.max(1)
            acc = 100. * predicted.eq(targets).sum().item() / targets.size(0)
            log.info(f"Batch {i}: Loss {loss.item():.4f}, Acc {acc:.2f}%")
            
    # Final check
    correct = 0
    total = 0
    student.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    log.info(f"Student 1-Epoch CE Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
