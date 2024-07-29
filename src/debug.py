import torch
import torch.nn as nn
import torch.nn.functional as F
class DebugTikhonovLayer(nn.Module):
    def __init__(self, num_classes):
        super(DebugTikhonovLayer, self).__init__()
        self.num_classes = num_classes
        self.lambda_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        print(f"Input x shape: {x.shape}")
        
        A = x
        print(f"A shape: {A.shape}")
        
        ATA = torch.matmul(A.t(), A)
        print(f"ATA shape: {ATA.shape}")
        
        I = torch.eye(self.num_classes, device=x.device)
        print(f"I shape: {I.shape}")
        
        print(f"lambda_param: {self.lambda_param}")
        
        try:
            P = torch.inverse(ATA + self.lambda_param * I)
            print(f"P shape: {P.shape}")
        except RuntimeError as e:
            print(f"Error in inverse operation: {e}")
            print(f"ATA + lambda * I shape: {(ATA + self.lambda_param * I).shape}")
            
        try:
            H = torch.matmul(A, torch.matmul(P, A.t()))
            print(f"H shape: {H.shape}")
        except NameError:
            print("H calculation skipped due to earlier error")
        
        return A  # Temporarily return A instead of H for debugging

class DebugSimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DebugSimpleCNN, self).__init__()
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
        self.tikhonov = DebugTikhonovLayer(num_classes)

    def forward(self, x):
        print(f"Input to CNN shape: {x.shape}")
        x = self.features(x)
        print(f"After features shape: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"After flatten shape: {x.shape}")
        x = nn.functional.relu(self.fc1(x))
        print(f"After fc1 shape: {x.shape}")
        x = self.fc2(x)
        print(f"After fc2 shape: {x.shape}")
        return self.tikhonov(x)

def debug_mlr_loss(H, features, targets, permuted_targets):
    print("\nDEBUG: MLR Loss Calculation")
    print(f"H shape: {H.shape}")
    print(f"features shape: {features.shape}")
    print(f"targets shape: {targets.shape}")
    print(f"Number of permuted targets: {len(permuted_targets)}")
    if len(permuted_targets) > 0:
        print(f"First permuted target shape: {permuted_targets[0].shape}")
    
    batch_size, num_classes = features.shape
    print(f"batch_size: {batch_size}, num_classes: {num_classes}")
    
    # Create noisy versions of targets
    noise_scale = 0.1  # Adjust this value as needed
    epsilon = noise_scale * torch.randn_like(targets)
    Y_plus = 0.5 * (targets + epsilon)
    Y_minus = 0.5 * (targets - epsilon)
    
    print(f"Epsilon stats: min={epsilon.min().item():.4f}, max={epsilon.max().item():.4f}, mean={epsilon.mean().item():.4f}, std={epsilon.std().item():.4f}")
    print(f"Y_plus stats: min={Y_plus.min().item():.4f}, max={Y_plus.max().item():.4f}, mean={Y_plus.mean().item():.4f}")
    print(f"Y_minus stats: min={Y_minus.min().item():.4f}, max={Y_minus.max().item():.4f}, mean={Y_minus.mean().item():.4f}")


    # Compute Y_tilde
    Y_tilde_plus = torch.matmul(H, Y_plus)
    Y_tilde_minus = torch.matmul(H, Y_minus)
    print(f"Y_tilde_plus shape: {Y_tilde_plus.shape}")
    print(f"Y_tilde_minus shape: {Y_tilde_minus.shape}")

    # Compute CE loss for true targets
    ce_loss_true = F.cross_entropy(Y_tilde_plus + Y_minus, targets.argmax(dim=1)) + \
                   F.cross_entropy(Y_tilde_minus + Y_plus, targets.argmax(dim=1))
    print(f"CE loss for true targets: {ce_loss_true.item()}")
    
    # If no permuted targets, return only the true targets loss
    if len(permuted_targets) == 0:
        print(f"Final loss (true targets only): {ce_loss_true.item()}")
        return ce_loss_true

    # Compute CE loss for permuted targets
    ce_loss_permuted = 0
    for i, perm_target in enumerate(permuted_targets):
        ce_loss_perm = F.cross_entropy(Y_tilde_plus + perm_target, perm_target.argmax(dim=1)) + \
                       F.cross_entropy(Y_tilde_minus + perm_target, perm_target.argmax(dim=1))
        ce_loss_permuted += ce_loss_perm
        print(f"CE loss for permuted target {i}: {ce_loss_perm.item()}")
    ce_loss_permuted /= len(permuted_targets)
    print(f"Average CE loss for permuted targets: {ce_loss_permuted.item()}")

    mlr_loss = ce_loss_true - ce_loss_permuted
    print(f"Final MLR loss: {mlr_loss.item()}")

    # Additional checks
    print(f"Max value in H: {H.max().item()}, Min value in H: {H.min().item()}")
    print(f"Max value in Y_tilde_plus: {Y_tilde_plus.max().item()}, Min value in Y_tilde_plus: {Y_tilde_plus.min().item()}")
    print(f"Max value in features: {features.max().item()}, Min value in features: {features.min().item()}")

    return mlr_loss

if __name__ == "__main__":
    # Test the model
    model = DebugSimpleCNN()
    test_input = torch.randn(64, 3, 32, 32)  # Assuming CIFAR-10 input size
    output = model(test_input)
    print(f"Final output shape: {output.shape}")