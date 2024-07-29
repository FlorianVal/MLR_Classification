import torch
import torch.nn as nn
import torch.nn.functional as F

def MLR_Loss(H, features, targets, permuted_targets):
    batch_size, num_classes = features.shape
    
    # Create noisy versions of targets
    noise_scale = 0.1  # Adjust this value as needed
    epsilon = noise_scale * torch.randn_like(targets)
    Y_plus = 0.5 * (targets + epsilon)
    Y_minus = 0.5 * (targets - epsilon)

    # Compute Y_tilde
    Y_tilde_plus = torch.matmul(H, Y_plus)  # Shape: (batch_size, num_classes)
    Y_tilde_minus = torch.matmul(H, Y_minus)  # Shape: (batch_size, num_classes)

    # Compute CE loss for true targets
    ce_loss_true = F.cross_entropy(Y_tilde_plus + Y_minus, targets.argmax(dim=1)) + \
                   F.cross_entropy(Y_tilde_minus + Y_plus, targets.argmax(dim=1))
   
    # If no permuted targets, return only the true targets loss
    if len(permuted_targets) == 0:
        return ce_loss_true

    # Compute CE loss for permuted targets
    ce_loss_permuted = 0
    for perm_target in permuted_targets:
        ce_loss_permuted += F.cross_entropy(Y_tilde_plus + perm_target, perm_target.argmax(dim=1)) + \
                            F.cross_entropy(Y_tilde_minus + perm_target, perm_target.argmax(dim=1))
    ce_loss_permuted /= len(permuted_targets)

    return ce_loss_true - ce_loss_permuted