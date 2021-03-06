import importlib
import logging
import math
import random
import torch
from tqdm import tqdm

from src.runner.utils import EpochLog

LOGGER = logging.getLogger(__name__.split('.')[-1])


class BaseTrainer:
    """The base class for all trainers.
    Args:
        device (torch.device): The device.
        train_dataloader (Dataloader): The training dataloader.
        valid_dataloader (Dataloader): The validation dataloader.
        net (BaseNet): The network architecture.
        loss_fns (LossFns): The loss functions.
        loss_weights (LossWeights): The corresponding weights of loss functions.
        metric_fns (MetricFns): The metric functions.
        optimizer (torch.optim.Optimizer): The algorithm to train the network.
        lr_scheduler (torch.optim.lr_scheduler): The scheduler to adjust the learning rate.
        logger (BaseLogger): The object for recording the log information.
        monitor (Monitor): The object to determine whether to save the checkpoint.
        num_epochs (int): The total number of training epochs.
        valid_freq (int): The validation frequency (default: 1).
        opt_level (str): The optimization level of apex.amp (default: 'O1').
    """

    def __init__(self, saved_dir, device, train_dataloader, valid_dataloader,
                 net, loss_fns, loss_weights, metric_fns, optimizer, lr_scheduler,
                 logger, monitor, num_epochs, valid_freq=1, freeze_param=False, unfreeze_epoch=1):
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.net = net
        self.loss_fns = loss_fns
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.metric_fns = metric_fns
        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        self.logger = logger
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.valid_freq = valid_freq
        self.epoch = 1
        self.freeze_param = freeze_param
        self.unfreeze_epoch = unfreeze_epoch


    def train(self):
        """The training process.
        """
        # freeze or not
        if self.freeze_param == True:
            self.net._freeze()
        # begin training
        while self.epoch <= self.num_epochs:
            # Do training and validation.
            LOGGER.info(f'Epoch {self.epoch}.')
            train_log, train_batch, train_outputs = self._run_epoch('train')
            LOGGER.info(f'Train log: {train_log}.')
            if self.epoch % self.valid_freq == 0:
                valid_log, valid_batch, valid_outputs = self._run_epoch('valid')
                LOGGER.info(f'Valid log: {valid_log}.')
            else:
                valid_log, valid_batch, valid_outputs = None, None, None

            # Adjust the learning rate.
            if self.lr_scheduler is None:
                pass
            else:
                self.lr_scheduler.step()

            # Record the log information.
            self.logger.write(self.epoch, train_log, train_batch, train_outputs,
                              valid_log, valid_batch, valid_outputs)

            if self.epoch % self.monitor.valid_freq == 0:
                # Save the best checkpoint.
                saved_path = self.monitor.is_best(self.epoch, valid_log)
                if saved_path is not None:
                    LOGGER.info(f'Save the best checkpoint to {saved_path} '
                                f'({self.monitor.mode} {self.monitor.target}: {self.monitor.best_score}).')
                    self.save(saved_path)
                else:
                    LOGGER.info(f'The best checkpoint is remained at epoch {self.monitor.best_epoch} '
                                f'({self.monitor.mode} {self.monitor.target}: {self.monitor.best_score}).')

                # Save the regular checkpoint.
                saved_path = self.monitor.is_saved(self.epoch)
                if saved_path is not None:
                    LOGGER.info(f'Save the regular checkpoint to {saved_path}.')
                    self.save(saved_path)

                # Early stop.
                if self.monitor.is_early_stopped():
                    LOGGER.info('Early stopped.')
                    break
            else:
                # Save the regular checkpoint.
                saved_path = self.monitor.is_saved(self.epoch)
                if saved_path is not None:
                    LOGGER.info(f'Save the regular checkpoint to {saved_path}.')
                    self.save(saved_path)

            # unfreeze or not
            if self.freeze_param == True and self.epoch == self.unfreeze_epoch:
               self.net._unfreeze()
               self.freeze_param = False
                
             # update epoch   
            self.epoch += 1

        self.logger.close()

    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('train' or 'valid').
        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        if mode == 'train':
            self.net.train()
            dataloader = self.train_dataloader
            dataloader_iterator = iter(dataloader)
            outter_pbar = tqdm(total=len(dataloader),
                               desc=mode,
                               ascii=True)

            epoch_log = EpochLog()
            for i in range(len(dataloader)):
                batch = next(dataloader_iterator)
                train_dict = self._train_step(batch)
                loss = train_dict['loss']
                losses = train_dict.get('losses')
                metrics = train_dict.get('metrics')
                outputs = train_dict.get('outputs')
                lr = self.optimizer.param_groups[0]['lr']

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # if isinstance(self.lr_scheduler, (CyclicLR, OneCycleLR)):
                #     self.lr_scheduler.step()
                # elif isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
                #     self.lr_scheduler.step((self.epoch - 1) + i / len(dataloader))

                if (i + 1) == len(dataloader) and not dataloader.drop_last:
                    batch_size = len(dataloader.dataset) % dataloader.batch_size
                else:
                    batch_size = dataloader.batch_size

                epoch_log.update(batch_size, loss, losses, metrics, lr)

                outter_pbar.update()
                outter_pbar.set_postfix(**epoch_log.on_step_end_log)

            outter_pbar.close()

        else:
            self.net.eval()
            dataloader = self.valid_dataloader
            pbar = tqdm(dataloader, desc=mode, ascii=True)

            epoch_log = EpochLog()
            for i, batch in enumerate(pbar):
                with torch.no_grad():
                    valid_dict = self._valid_step(batch)
                    loss = valid_dict['loss']
                    losses = valid_dict.get('losses')
                    metrics = valid_dict.get('metrics')
                    outputs = valid_dict.get('outputs')

                if (i + 1) == len(dataloader) and not dataloader.drop_last:
                    batch_size = len(dataloader.dataset) % dataloader.batch_size
                else:
                    batch_size = dataloader.batch_size

                epoch_log.update(batch_size, loss, losses, metrics)

                pbar.set_postfix(**epoch_log.on_step_end_log)

        return epoch_log.on_epoch_end_log, batch, outputs

    def _train_step(self, batch):
        """The user-defined training logic.
        Args:
            batch (dict or sequence): A batch of the data.
        Returns:
            train_dict (dict): The computed results.
                train_dict['loss'] (torch.Tensor)
                train_dict['losses'] (dict, optional)
                train_dict['metrics'] (dict, optional)
                train_dict['outputs'] (dict, optional)
        """
        raise NotImplementedError

    def _valid_step(self, batch):
        """The user-defined validation logic.
        Args:
            batch (dict or sequence): A batch of the data.
        Returns:
            valid_dict (dict): The computed results.
                valid_dict['loss'] (torch.Tensor)
                valid_dict['losses'] (dict, optional)
                valid_dict['metrics'] (dict, optional)
                valid_dict['outputs'] (dict, optional)
        """
        raise NotImplementedError

    def save(self, path):
        """Save the model checkpoint.
        Args:
            path (Path): The path to save the model checkpoint.
        """
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'monitor': self.monitor.state_dict(),
            'epoch': self.epoch,
            'random_state': random.getstate(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.get_rng_state_all()
        }, path)

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.lr_scheduler is not None and checkpoint['lr_scheduler'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        elif self.lr_scheduler is None and checkpoint['lr_scheduler'] is not None:
            LOGGER.warning('The learning rate scheduler is no longer in use.')
        elif self.lr_scheduler is not None and checkpoint['lr_scheduler'] is None:
            LOGGER.warning('Start using the learning rate scheduler.')

        self.monitor.load_state_dict(checkpoint['monitor'])
        self.epoch = checkpoint['epoch'] + 1
        random.setstate(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        torch.cuda.set_rng_state_all(checkpoint['torch_cuda_random_state'])
