import torch
import torch.nn as nn

__all__ = [
    'Accuracy',
]


class Accuracy(nn.Module):
    """The accuracy for the classification task.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N, 1): The data target.
        Returns:
            metric (torch.Tensor) (0): The accuracy.
        """
        pred = output.argmax(dim=1, keepdim=False)
        return (pred == target).float().mean()

    def get_name(self):
        return 'Accuracy'
