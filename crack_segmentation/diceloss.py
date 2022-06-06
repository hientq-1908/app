import torch
import torch.nn as nn
from sys import exit

class DiceLoss(nn.Module):
    """
    Multi-class soft dice loss for multi class sementic segmentation
    """
    def __init__(self, epsilon=1e-4) -> None:
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        """
        y_pred, y_true: N_class, dim, dim
        """
        assert y_pred.shape == y_true.shape
        if len(y_pred.shape) == 2:
            # 1 single sample
            y_true = y_true.unsqueeze(0)
            y_pred = y_pred.unsqueeze(0)

        numerator = 2 * (y_pred * y_true).sum(dim=(-1,-2)) + self.epsilon
        dinominator = (y_pred**2).sum(dim=(-1,-2)) + (y_true**2).sum(dim=(-1,-2)) + self.epsilon
        dice_loss = 1 - numerator / dinominator
        return torch.mean(dice_loss)
