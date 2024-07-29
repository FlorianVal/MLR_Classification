import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, lambda_param = 1.0):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.tikhonov = TikhonovLayer(512, lambda_param)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        features = self.fc2(x)  # Shape: (batch_size, num_classes)
        H = self.tikhonov(x)  # Shape: (batch_size, batch_size)
        return H, features

class TikhonovLayer(nn.Module):
    def __init__(self, feature_size, lambda_param = 1.0):
        super(TikhonovLayer, self).__init__()
        self.feature_size = feature_size
        self.lambda_param = nn.Parameter(torch.tensor(lambda_param))

    def forward(self, x):
        A = x  # Shape: (batch_size, feature_size)
        ATA = torch.matmul(A.t(), A)  # Shape: (feature_size, feature_size)
        I = torch.eye(self.feature_size, device=x.device)  # Shape: (feature_size, feature_size)
        P = torch.inverse(ATA + self.lambda_param * I)  # Shape: (feature_size, feature_size)
        H = torch.matmul(A, torch.matmul(P, A.t()))  # Shape: (batch_size, batch_size)
        return H