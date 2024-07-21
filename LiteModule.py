from abc import abstractmethod
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn

from LightningLite.Logger import Logger


class LiteModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.current_epoch = 0
        self.current_step = 0
        self.device = 'cpu'
        # attributes that you don't want to distribute to mutiple devices
        self.distribution_ignore = []
        self.cuda_ignore = []  # attributes that you don't want to move to cuda
        self.logger = None

    def save(self, save_folder):
        torch.save(self.state_dict(), f'{save_folder}/model.pt')

    def load(self, path):
        state_dict = torch.load(path)
        # make the state_dict compatible for nn.DataParallel
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}
        self.load_state_dict(state_dict)

    '''training'''
    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def on_train_batch_end(self, batch_result):
        pass

    @abstractmethod
    def on_train_epoch_end(self, training_results: Optional[List] = None, val_results: Optional[List] = None):
        pass

    @abstractmethod
    def on_train_end(self, results: Optional[List] = None):
        pass

    '''validation'''
    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def on_validation_batch_end(self, batch_result):
        pass

    @abstractmethod
    def on_validation_epoch_end(self, training_results: Optional[List] = None, val_results: Optional[List] = None):
        pass

    @abstractmethod
    def on_validation_end(self, results: Optional[List] = None):
        pass

    '''test'''
    @abstractmethod
    def test_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def on_test_batch_end(self, batch_result):
        pass

    @abstractmethod
    def on_test_epoch_end(self, training_results: Optional[List] = None, val_results: Optional[List] = None):
        pass

    @abstractmethod
    def on_test_end(self, results: Optional[List] = None):
        pass

    '''predict'''
    @abstractmethod
    def predict_step(self, batch):
        pass

    def log_dict(self, dict: Dict, on_step=False, on_epoch=True, prog_bar=True):
        if on_epoch:
            self.logger.add_epoch_log(dict)
        if on_step:
            self.logger.add_log(dict, self.current_epoch, self.current_step)
