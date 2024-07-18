import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.model import SimpleCNN
from src.utils import get_device

def train(args):
    device = get_device()
    print(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = args.model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(f'runs/cifar10_experiment_{args.exp_name}')

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                writer.add_scalar('training loss', running_loss / 200, epoch * len(trainloader) + i)
                running_loss = 0.0

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('validation loss', val_loss / len(testloader), epoch)
        writer.add_scalar('validation accuracy', accuracy, epoch)
        print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy}%')

    print('Finished Training')
    writer.close()
    
def main():
    parser = argparse.ArgumentParser(description='CIFAR10 Training')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--exp-name', type=str, default='default', help='experiment name')
    args = parser.parse_args()

    args.model = SimpleCNN  # Add the model to args

    train(args)

if __name__ == '__main__':
    main()