import torch.nn.functional as F
import torch
from src.runner.trainers import BaseTrainer


class MangoTrainer(BaseTrainer):
    """
    """

    def __init__(self, meta_data=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_data = meta_data

    def _train_step(self, batch):
        inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)
        outputs = self.net(inputs)
        targets = targets.squeeze(1)
        losses = {loss.get_name():loss(outputs, targets) for loss in self.loss_fns}
        loss = 0
        for i, loss_name in enumerate(losses.keys()):
            loss += losses[loss_name] * self.loss_weights[i]

        metrics = {metric.get_name():metric(outputs, targets) for metric in self.metric_fns}

        # _, preds = outputs.max(1)
        # correct = preds.eq(targets).sum()
        # print(correct.float() / outputs.size(0))
        # preds = outputs.argmax(dim=1, keepdim=True)
        # print((preds == targets).float().mean())

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': outputs
        }

    def _valid_step(self, batch):
        inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)
        outputs = self.net(inputs)
        targets = targets.squeeze(1)
        losses = {loss.get_name():loss(outputs, targets) for loss in self.loss_fns}
        loss = 0
        for i, loss_name in enumerate(losses.keys()):
            loss += losses[loss_name] * self.loss_weights[i]

        metrics = {metric.get_name():metric(outputs, targets) for metric in self.metric_fns}

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': outputs
        }