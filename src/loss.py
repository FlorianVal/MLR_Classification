import torch
import torch.nn as nn
import torch.nn.functional as F

def MLR_Loss(H, preds, targets, permuted_targets):
    batch_size, num_classes = preds.shape
    
    # Create noisy versions of targets
    noise_scale = 0.1  # Adjust this value as needed
    epsilon = noise_scale * torch.randn_like(targets)
    
    Y_plus = 0.5 * (targets + epsilon)
    Y_minus = 0.5 * (targets - epsilon)

    # Compute Y_tilde
    Y_tilde_plus = torch.matmul(H, Y_plus)  # Shape: (batch_size, num_classes)
    Y_tilde_minus = torch.matmul(H, Y_minus)  # Shape: (batch_size, num_classes)

    # Compute CE loss for true targets
    ce_loss_true = torch.mean(
                   F.cross_entropy(torch.softmax(Y_tilde_plus + Y_minus, dim=-1), targets.argmax(dim=1), reduction='none') + \
                   F.cross_entropy(torch.softmax(Y_tilde_minus + Y_plus, dim=1), targets.argmax(dim=1), reduction='none'))
    # If no permuted targets, return only the true targets loss
    if len(permuted_targets) == 0:
        return ce_loss_true

    ce_loss_permuted = 0
    for perm_target in permuted_targets:

        xi = noise_scale * torch.randn_like(targets)
        
        # Compute CE loss for permuted targets
        pi_plus = 0.5 * (perm_target + xi)
        pi_minus = 0.5 * (perm_target - xi)

        # Compute Y_tilde
        pi_tilde_plus = torch.matmul(H, pi_plus)  # Shape: (batch_size, num_classes)
        pi_tilde_minus = torch.matmul(H, pi_minus)  # Shape: (batch_size, num_classes)
        
        
        ce_loss_permuted += F.cross_entropy(torch.softmax(pi_tilde_plus + pi_minus, dim=-1), perm_target.argmax(dim=1)) + \
                            F.cross_entropy(torch.softmax(pi_tilde_minus + pi_plus, dim=-1), perm_target.argmax(dim=1))
    ce_loss_permuted /= len(permuted_targets)
    return ce_loss_true - ce_loss_permuted