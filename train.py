import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from src.model import SimpleCNN
from src.utils import get_device
from src.loss import MLR_Loss
from src.dataset import PermutedDataset, collate_permuted
from src.debug import debug_mlr_loss
from tqdm import tqdm
from clearml import Task, Logger


def train(args, task):
    
    logger = Logger.current_logger()

    device = get_device()
    print(f"Using device: {device}") 
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset_permuted = PermutedDataset(trainset, args.num_permutations)
    trainloader = DataLoader(trainset_permuted, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_permuted)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = args.model(lambda_param=args.lambda_param).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    task.connect(model)
    task.connect(optimizer)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct_H = 0
        correct_features = 0
        total = 0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for i, data in pbar:  
            inputs, labels, *labels_permuted = [d.to(device) for d in data]

            optimizer.zero_grad()

            # Create one-hot encoded labels
            targets = nn.functional.one_hot(labels, num_classes=num_classes).float()
            permuted_targets = [nn.functional.one_hot(lp, num_classes=num_classes).float() for lp in labels_permuted]

            H, features = model(inputs, targets)

            # Compute MLR loss
            loss = MLR_Loss(H, features, targets, permuted_targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracies
            predicted_H = torch.softmax(torch.matmul(H, targets), dim=-1).argmax(dim=-1)
            # print(f"{H.tolist()}, {targets.tolist()}, {torch.softmax(torch.matmul(H, targets), dim=-1).tolist()}")
            predicted_features = torch.softmax(features, dim=-1).argmax(dim=-1)
            total += labels.size(0)
            correct_H += (predicted_H == labels).sum().item()
            correct_features += (predicted_features == labels).sum().item()
            accuracy_H = 100 * correct_H / total
            accuracy_features = 100 * correct_features / total

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc_H': f'{accuracy_H:.2f}%', 
                'acc_feat': f'{accuracy_features:.2f}%'
            })
            # log every 1/10th of the epoch
            if (i + 1) % (len(trainloader) // 10) == 0:
                logger.report_scalar(title='Loss', series='train', value=running_loss / 200, iteration=epoch * len(trainloader) + i)
                logger.report_scalar(title='Accuracy', series='train_H', value=accuracy_H, iteration=epoch * len(trainloader) + i)
                logger.report_scalar(title='Accuracy', series='train_features', value=accuracy_features, iteration=epoch * len(trainloader) + i)

                running_loss = 0.0
                correct_H = 0
                correct_features = 0
                total = 0

        # Validation
        model.eval()
        correct_H = 0
        correct_features = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                H, features = model(images)
                targets = nn.functional.one_hot(labels, num_classes=num_classes).float()
                predicted_H = torch.softmax(torch.matmul(H, targets), dim=-1).argmax(dim=-1)
                predicted_features = torch.softmax(features, dim=-1).argmax(dim=-1)
                total += labels.size(0)
                correct_H += (predicted_H == labels).sum().item()
                correct_features += (predicted_features == labels).sum().item()

        val_accuracy_H = 100 * correct_H / total
        val_accuracy_features = 100 * correct_features / total
        logger.report_scalar(title='Accuracy', series='val_H', value=val_accuracy_H, iteration=(epoch+1) * len(trainloader))
        logger.report_scalar(title='Accuracy', series='val_features', value=val_accuracy_features, iteration=(epoch+1) * len(trainloader))
        print(f'Epoch {epoch+1}, Accuracy_H: {val_accuracy_H:.2f}%, Accuracy_features: {val_accuracy_features:.2f}%')
    
    return model
    print('Finished Training')
    
def main():
    Task.set_credentials(
        api_host='http://localhost:8008',
        web_host='http://localhost:8080',
        files_host='http://localhost:8081',
    )

    parser = argparse.ArgumentParser(description='CIFAR10 Training with MLR Loss')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--num-epochs', type=int, default=25, help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--exp-name', type=str, default='mlr', help='experiment name')
    parser.add_argument('--num-permutations', type=int, default=5, help='number of permuted datasets (default: 5)')
    parser.add_argument('--lambda_param' , type=float, default=0.1, help='weight for the thikonov regularization term (default: 0.1)')
    args = parser.parse_args()

    task = Task.init(project_name="MLR Classification", task_name=f"MLR_training_{time.strftime('%Y%m%d-%H%M%S')}_{args.exp_name}")

    args.model = SimpleCNN  # Add the model to args
    task.connect(args)
    train(args, task)

if __name__ == '__main__':
    main()