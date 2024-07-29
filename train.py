import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.model import SimpleCNN
from src.utils import get_device
from src.loss import MLR_Loss
from src.dataset import PermutedDataset, collate_permuted
from src.debug import debug_mlr_loss

def train(args):
    device = get_device()
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset_permuted = PermutedDataset(trainset, args.num_permutations)
    trainloader = DataLoader(trainset_permuted, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_permuted)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = args.model(lambda_param=args.lambda_param).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(f'runs/cifar10_experiment_{args.exp_name}_{time.time()}')

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, *labels_permuted = [d.to(device) for d in data]

            optimizer.zero_grad()
            H, features = model(inputs)

            # Create one-hot encoded labels
            targets = nn.functional.one_hot(labels, num_classes=10).float()
            permuted_targets = [nn.functional.one_hot(lp, num_classes=10).float() for lp in labels_permuted]

            # Compute MLR loss
            # if i < 5:
            #     loss = debug_mlr_loss(H, features, targets, permuted_targets)
            # else:
            loss = MLR_Loss(H, features, targets, permuted_targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                writer.add_scalar('training loss', running_loss / 200, epoch * len(trainloader) + i)
                running_loss = 0.0

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                _, features = model(images)
                predicted = features.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('validation accuracy', accuracy, epoch)
        print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy}%')



    print('Finished Training')
    writer.close()
    
def main():
    parser = argparse.ArgumentParser(description='CIFAR10 Training with MLR Loss')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--exp-name', type=str, default='mlr', help='experiment name')
    parser.add_argument('--num-permutations', type=int, default=5, help='number of permuted datasets (default: 5)')
    parser.add_argument('--lambda_param' , type=float, default=0.1, help='weight for the thikonov regularization term (default: 0.1)')
    args = parser.parse_args()

    args.model = SimpleCNN  # Add the model to args

    train(args)

if __name__ == '__main__':
    main()