import logging
import torch
import csv
from tqdm import tqdm

from src.runner.utils import EpochLog

LOGGER = logging.getLogger(__name__.split('.')[-1])


class BasePredictor:
    """The base class for all predictors.
    Args:
        saved_dir (Path): The root directory of the saved data.
        device (torch.device): The device.
        test_dataloader (Dataloader): The testing dataloader.
        net (BaseNet): The network architecture.
        loss_fns (LossFns): The loss functions.
        loss_weights (LossWeights): The corresponding weights of loss functions.
        metric_fns (MetricFns): The metric functions.
    """

    def __init__(self, saved_dir, device, test_dataloader, net, output_csv_path):
        self.saved_dir = saved_dir
        self.device = device
        self.test_dataloader = test_dataloader
        self.net = net.to(device)
        self.output_csv_path = output_csv_path

    def predict(self):
        """The testing process.
        """
        self.net.eval()
        dataloader = self.test_dataloader
        pbar = tqdm(dataloader, desc='test', ascii=True)

        predictions = []
        for i, batch in enumerate(pbar):
            with torch.no_grad():
                test_dict = self._test_step(batch)
                cls_output = test_dict['cls_output']
                file_name = test_dict['file_name']
                predictions.append((file_name, cls_output))

        with open(self.output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_id', 'label'])
            for i, item in enumerate(predictions):
                file_name, pred = item
                writer.writerow([file_name, pred])

    def _test_step(self, batch):
        """The user-defined testing logic.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            test_dict (dict): The computed results.
                test_dict['loss'] (torch.Tensor)
                test_dict['losses'] (dict, optional)
                test_dict['metrics'] (dict, optional)
        """
        raise NotImplementedError

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.net.load_state_dict(checkpoint['net'])
