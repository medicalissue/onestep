import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from distill.models import ResNetWrapper
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 2. Model (Pretrained=False to start from scratch, or True and undertrain?)
    # Starting from scratch for 1 epoch is good for "Weak" (random init -> ~40-50% in 1 epoch).
    # Or use Pretrained and corrupt it? 
    # Let's use Pretrained=False and train for 5 epochs to get a "decent but weak" teacher (~60-70%).
    # Actually, 1 epoch from scratch might be too weak (<40%).
    # Let's try Pretrained=True (ImageNet) and finetune for just 0.5 epoch or 1 epoch on CIFAR.
    # ImageNet ResNet on CIFAR without finetuning is bad.
    # Let's do: Pretrained=False, Train for 5 epochs.
    
    log.info("Initializing Teacher (ResNet18) from scratch...")
    teacher = ResNetWrapper(num_classes=10, pretrained=False).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=1e-3)
    
    epochs = 2 # 2 epochs should give ~50-60% accuracy
    
    for epoch in range(epochs):
        teacher.train()
        total_loss = 0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if i % 100 == 0:
                log.info(f"Epoch [{epoch+1}/{epochs}] Batch [{i}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
                
        # Validation
        teacher.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = teacher(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        acc = 100.*test_correct/test_total
        log.info(f"Epoch {epoch+1} Test Accuracy: {acc:.2f}%")

    torch.save(teacher.state_dict(), "weak_teacher.pth")
    log.info(f"Weak Teacher saved to weak_teacher.pth (Acc: {acc:.2f}%)")

if __name__ == "__main__":
    main()
