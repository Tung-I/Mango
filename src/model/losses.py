import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'BCELossWrapper',
    'CrossEntropyLossWrapper',
]


class BCELossWrapper(nn.Module):
    """The binary cross-entropy loss wrapper which combines torch.nn.BCEWithLogitsLoss (with logits)
    and torch.nn.BCELoss (with probability).

    Ref: 
        https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
        https://pytorch.org/docs/stable/nn.html#bceloss

    Args:
        with_logits (bool, optional): Specify the output is logits or probability (default: True).
        weight (sequence, optional): The same argument in torch.nn.BCEWithLogitsLoss and torch.nn.BCELoss
            but its type is sequence for the configuration purpose (default: None).
        pos_weight (sequence, optional): The same argument in torch.nn.BCEWithLogitsLoss
            but its type is sequence for the configuration purpose (default: None).
    """

    def __init__(self, with_logits=True, weight=None, pos_weight=None, **kwargs):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
            kwargs.update(weight=weight)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
            kwargs.update(pos_weight=pos_weight)
        self.loss_fn = (nn.BCEWithLogitsLoss if with_logits else nn.BCELoss)(**kwargs)

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, *): The output logits or probability.
            target (torch.LongTensor) (N, *): The target where each value is 0 or 1.

        Returns:
            loss (torch.Tensor) (0): The binary cross entropy loss.
        """
        return self.loss_fn(output, target)
    
    def get_name(self):
        return 'BCELoss'


class CrossEntropyLossWrapper(nn.Module):
    """The cross-entropy loss wrapper which combines torch.nn.CrossEntropyLoss (with logits)
    and torch.nn.NLLLoss (with probability).

    Ref: 
        https://pytorch.org/docs/stable/nn.html#crossentropyloss
        https://pytorch.org/docs/stable/nn.html#nllloss

    Args:
        with_logits (bool, optional): Specify the output is logits or probability (default: True).
        weight (sequence, optional): The same argument in torch.nn.CrossEntropyLoss and torch.nn.NLLLoss
            but its type is sequence for the configuration purpose (default: None).
    """

    def __init__(self, with_logits=True, weight=None, **kwargs):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
            kwargs.update(weight=weight)
        self.loss_fn = (nn.CrossEntropyLoss if with_logits else nn.NLLLoss)(**kwargs)

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The output logits or probability.
            target (torch.LongTensor) (N, *): The target where each value is between 0 and C-1.

        Returns:
            loss (torch.Tensor) (0): The cross entropy loss.
        """
        return self.loss_fn(output, target)

    def get_name(self):
        return 'CELoss'


