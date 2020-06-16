import torch
import nibabel as nib
import torch.nn.functional as F

from src.runner.predictors import BasePredictor


class MangoPredictor(BasePredictor):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.test_dataloader.batch_size != 1:
            raise ValueError(f'The testing batch size should be 1. Got {self.test_dataloader.batch_size}.')
        self.num_to_class = ['A', 'B', 'C']

    def _test_step(self, batch):
        input, file_name = batch['input'].to(self.device), batch['file_name']
        file_name = file_name[0]
        output = self.net(input)
        pred = output.argmax(dim=1, keepdim=False)
        cls_output = self.num_to_class[pred[0]]
        return {'cls_output': cls_output, 'file_name': file_name}