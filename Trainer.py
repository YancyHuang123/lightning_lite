
from datetime import datetime
import os
import time
import torch
import torch.nn as nn

from .Printer import Printer
from .LiteModule import LiteModule
from .Logger import Logger
from .Timer import Timer
from .Tools import to_device, model_distribute, create_folder


class Trainer():
    def __init__(self, max_epochs=0, accelerator='cpu', devices='cpu', distribution=False, log_every_n_steps=50, disable_output=False, experiment_floder='lite_logs', cur_exper_folder=None, log_name='log.csv') -> None:
        '''
        Trainer manages three fundermetal elements to a deep learning experiment: 
        1. timer: gives you the full control of how long the experiment has taken and would take
        2. printer: tells you the real-time info of current experiment
        3. logger: stores full info of the experiment for later review and ploting
        '''

        # TODO: cancel keeping results from step returns, leaving user to define in-class variables instead.

        self.max_epochs = max_epochs
        self.device = devices
        self.accelerator = accelerator
        self.experiment_folder = experiment_floder  # folder keeps all experiments
        self.cur_exper_folder = cur_exper_folder  # folder keeps current experiments
        self.step_idx = 0
        self.log_every_n_steps = log_every_n_steps
        self.distribution = distribution
        self.disable_output = disable_output

        self.cur_exper_folder_path = create_folder(
            experiment_floder, cur_exper_folder)
        self.logger = Logger(self.cur_exper_folder_path, log_name=log_name)
        self.timer = Timer()
        self.printer = Printer(log_every_n_steps, max_epochs, disable_output)

    def fit(self, model: LiteModule, train_loader, val_loader=[]):
        self._stage_start_process(model, 'train')

        for epoch_idx in range(self.max_epochs):
            model.current_epoch = epoch_idx
            self.timer.epoch_start()

            train_epoch_results = []
            train_epoch_results.append(self._epoch_step(
                model, epoch_idx, train_loader, 'train'))

            val_epoch_results = []
            train_epoch_results.append(self._epoch_step(
                model, epoch_idx, val_loader, 'validation'))

            model.on_train_epoch_end(train_epoch_results, val_epoch_results)
            self._epoch_end_process(model, epoch_idx)

        self._stage_end_process(model, 'train')

    def test(self, model, test_loader):
        self._stage_start_process(model, 'test')
        for epoch_idx in range(1):
            model.current_epoch = epoch_idx
            self.timer.epoch_start()

            test_epoch_results = []
            test_epoch_results.append(self._epoch_step(model, epoch_idx, test_loader,
                                                       'test'))

            model.on_test_epoch_end(test_epoch_results)

            self._epoch_end_process(model, epoch_idx)

        self._stage_end_process(model, 'test')

    def predict(self, model, X):
        self._stage_start_process(model, 'prediction')
        pred_result = model.predict_step(X)
        return pred_result

    def _stage_start_process(self, model, stage):
        # distribute model to accelerators
        model = model_distribute(model, self.accelerator, self.distribution)
        model.logger = self.logger
        self.timer.stage_start()
        self.printer.stage_start_output(stage)

    def _epoch_step(self, model, epoch_idx, dataset, stage):
        if stage != 'train':
            torch.set_grad_enabled(False)
            model.eval()
        else:
            torch.set_grad_enabled(True)
            model.train()

        dataset_len = len(dataset)
        epoch_results = []
        for batch_idx, batch in enumerate(dataset):
            batch_output = self._batch_step(
                model, epoch_idx, dataset_len, batch_idx, batch, stage)
            epoch_results.append(batch_output)

            # batch end hook
            getattr(model, f'on_{stage}_batch_end')(batch_output)

        torch.set_grad_enabled(True)

    def _batch_step(self, model, epoch_idx, dataset_len, batch_idx, batch, stage):
        batch = to_device(batch, model.device)
        # some stage's api names are different
        stage_ing = {'train': 'training',
                     'test': 'test', 'validation': 'validation'}
        # DO NOT return tensors directly, this can lead to gpu menory shortage !!
        result = getattr(model, f'{stage_ing[stage]}_step')(
            batch, batch_idx)
        self.step_idx += 1
        model.current_step = self.step_idx
        self.printer.batch_output(
            stage, epoch_idx, batch_idx, dataset_len, self.logger.last_log)
        return result

    def _epoch_end_process(self, model, epoch_idx):
        '''logging, printing, setting timer at the end of epoch'''
        self.logger.reduce_epoch_log(epoch_idx, self.step_idx)
        self.logger.save_log()
        model.save(self.cur_exper_folder_path)
        self.timer.epoch_end()
        self.printer.epoch_end_output(
            epoch_idx, self.timer.epoch_cost, self.logger.last_log)

    def _stage_end_process(self, model, stage):
        getattr(model, f'on_{stage}_end')()
        self.timer.stage_end()
        self.printer.stage_end_output(stage, self.timer.total_cost)
