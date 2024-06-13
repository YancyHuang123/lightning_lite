from abc import abstractmethod
from ast import Module
from datetime import datetime
import os
import time
import torch
import torch.nn as nn

from .Printer import Printer
from .Module import Module
from .Logger import Logger
from .Timer import Timer


class Trainer():
    def __init__(self, max_epochs, accelerator: str, devices=None, output_interval=50, log_folder='lite_logs', saving_folder=None, log_name='log.csv', distribution=False) -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.acceletator = accelerator
        self.devices = devices
        self.log_folder = log_folder  # folder keeps all training logs
        self.cur_log_folder = saving_folder  # folder keeps current training log
        self.step_idx = 0
        self.output_interval = output_interval
        self.distribution = distribution

        '''
        the three fundermetal elements to a deep learning experiment: 
        1. timer: gives you the full control of how long the experiment will take
        2. printer: gives you the real-time info of current experiment
        3. logger: gives you full info of the whole experiment for later review
        '''
        if saving_folder is None:
            self.create_saving_folder()
        self.logger = Logger(self.cur_log_folder, log_name=log_name)
        self.timer = Timer()
        self.printer = Printer(output_interval, max_epochs)

    def fit(self, model: Module, train_loader, val_loader=[]):
        # distribute model to accelerators
        model = self.model_distribute(model)
        model.logger = self.logger

        # epoch loop
        self.timer.stage_start()
        print('\n'+'>'*10+'Train started\n')
        for epoch_idx in range(self.max_epochs):
            model.current_epoch = epoch_idx
            self.timer.epoch_start()

            train_epoch_results = []
            train_epoch_results.append(self._epoch_step(
                model, epoch_idx, train_loader, 'train'))

            val_epoch_results = []
            train_epoch_results.append(self._epoch_step(
                model, epoch_idx, val_loader, 'validation'))

            # train epoch end hook
            model.on_train_epoch_end(train_epoch_results, val_epoch_results)

            self._epoch_end_process(model, epoch_idx)

        # train end hook
        model.on_train_end()
        self.timer.stage_end()
        self.printer.end_output('Traning', self.timer.total_cost)

    def test(self, model, test_loader):
        model = self.model_distribute(model)
        model.logger = self.logger

        # test start
        self.timer.stage_start()
        print('\n'+'>'*10+'Test started\n')
        for epoch_idx in range(self.max_epochs):
            model.current_epoch = epoch_idx
            self.timer.epoch_start()

            test_epoch_results = []
            test_epoch_results.append(self._epoch_step(model, epoch_idx, test_loader,
                                                       'test'))

            # test epoch end hook
            model.on_test_epoch_end(test_epoch_results)

            self._epoch_end_process(model, epoch_idx)

        # train end hook
        self._stage_end_process(model, 'test')

    def _stage_end_process(self, model, stage):
        model.on_test_end()
        self.timer.stage_end()
        self.printer.end_output(stage, self.timer.total_cost)

    def _epoch_end_process(self, model, epoch_idx):
        '''logging, printing, setting timer at the end of epoch'''
        self.logger.reduce_epoch_log(epoch_idx, self.step_idx)
        self.logger.save_log()
        model.save(self.cur_log_folder)
        self.timer.epoch_end()
        self.printer.epoch_output(
            epoch_idx, self.timer.epoch_cost, self.logger.last_log)

    def _epoch_step(self, model, epoch_idx, dataset, stage):
        if stage == 'train':
            torch.set_grad_enabled(False)
            model.eval()
        else:
            model.train()

        dataset_len = len(dataset)
        epoch_results = []
        for batch_idx, batch in enumerate(dataset):

            batch_output = self._batch_step(
                model, epoch_idx, dataset_len, batch_idx, batch, stage)
            epoch_results.append(batch_output)

            # batch end hook
            model.getattr(f'on_{stage}_batch_end')(batch_output)

        torch.set_grad_enabled(True)

    def _batch_step(self, model, epoch_idx, dataset_len, batch_idx, batch, stage):
        batch = self._to_device(batch, model.device)
        # DO NOT return tensors directly, this can lead to gpu menory shortage !!
        result = model.training_step(batch, batch_idx)
        self.step_idx += 1
        model.current_step = self.step_idx
        self.printer.batch_output(
            stage, epoch_idx, batch_idx, dataset_len, self.logger.last_log)
        return result

    def predict(self, model, predict_loader):
        model = self.model_distribute(model)
        model.logger = self.logger
        predictset_len = len(predict_loader)

        # test start
        self.timer.stage_start()
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nTest started\n')
        with torch.no_grad():
            model.eval()
            predict_results = []
            for batch_idx, batch in enumerate(predict_loader):
                batch = self._to_device(batch, model.device)
                # !!DO NOT return tensors directly, this can lead to gpu menory shortage !! use a.item() instead
                result = model.predict_step(batch, batch_idx)
                predict_results.append(result)
                self.printer.batch_output(
                    'Predicting', 0, batch_idx, predictset_len, self.logger.last_log)
            model.on_test_end(predict_results)
            self.logger.save_log()
            self.printer.epoch_output(
                0, 0, self.logger.last_log)
        # prediction end
        self.timer.stage_end()
        self.printer.end_output('Prediction', self.timer.total_cost)

    def _to_device(self, batch, device):
        ''' Move batch data to device automatically
            batch: should be either of the following forms -- tensor / [tensor1, tensor2,...] / [[tensor1,tensor2..],[tensor1,tensor2..],...]
            return the same form of batch data with all tensor on the dedicated device
        '''
        items = []
        for x in batch:
            if torch.is_tensor(x):
                items.append(x.to(device))
            elif isinstance(x, list):
                item = []
                for y in x:
                    item.append(y.to(device))
                items.append(item)
            else:
                raise Exception('outputs of dataloader unsupported')
        return tuple(items) if len(items) != 1 else items[0]

    def model_distribute(self, model: Module) -> Module:
        if self.acceletator == 'gpu':
            model.device = 'cuda'
            for key in model._modules:
                value = getattr(model, key)
                if key not in model.cuda_ignore:
                    if key not in model.distribution_ignore:
                        if self.distribution == False:
                            value = value.to('cuda')
                        else:
                            value = nn.DataParallel(value).to('cuda')
                    else:
                        value = value.to('cuda')
                    setattr(model, key, value)
        return model

    def create_saving_folder(self):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.log_folder
        os.makedirs(f'{folder}', exist_ok=True)
        os.mkdir(f"{folder}/{time}")
        self.cur_log_folder = f"{folder}/{time}"
