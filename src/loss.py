import torch
import torch.nn as nn

class TikhonovLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(TikhonovLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.lambda_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        A = x  # A_L-1
        ATA = torch.matmul(A.t(), A)
        I = torch.eye(self.in_features, device=x.device)
        P = torch.inverse(ATA + self.lambda_param * I)
        H = torch.matmul(A, torch.matmul(P, A.t()))
        return H

def MLR_Loss(outputs, targets, permuted_targets, tikhonov_layer):
    n, num_classes = outputs.shape
    
    # Create noisy versions of targets
    epsilon = torch.randn_like(targets).to(targets.device)
    Y_plus = 0.5 * (targets + epsilon)
    Y_minus = 0.5 * (targets - epsilon)

    # Apply Tikhonov operator
    H = tikhonov_layer(outputs)
    Y_tilde_plus = torch.matmul(H, Y_plus)
    Y_tilde_minus = torch.matmul(H, Y_minus)

    # Compute CE loss for true targets without reduction
    ce_loss_true = F.cross_entropy(Y_tilde_plus + Y_minus, targets.argmax(dim=1), reduction='none') + \
                   F.cross_entropy(Y_tilde_minus + Y_plus, targets.argmax(dim=1), reduction='none')

    # Compute CE loss for permuted targets without reduction
    ce_loss_permuted = torch.zeros_like(ce_loss_true)
    for perm_target in permuted_targets:
        ce_loss_permuted += F.cross_entropy(Y_tilde_plus + perm_target, perm_target.argmax(dim=1), reduction='none') + \
                            F.cross_entropy(Y_tilde_minus + perm_target, perm_target.argmax(dim=1), reduction='none')
    ce_loss_permuted /= len(permuted_targets)

    # Compute the final MLR loss
    mlr_loss = ce_loss_true.mean() - ce_loss_permuted.mean()

    return ce_loss_true - ce_loss_permuted