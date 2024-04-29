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
from .Timer import WrapperTimer


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
        self.timer = WrapperTimer()
        self.printer = Printer(output_interval, max_epochs)

    def fit(self, model: Module, train_loader, val_loader=[]):
        model = self.model_distribute(model)  # distribute model to accelerators
        model.logger = self.logger  # type:ignore
        trainset_len = len(train_loader)
        valset_len = len(val_loader)

        # epoch loop
        self.timer.training_start()
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nTraining started\n')
        for epoch_idx in range(self.max_epochs):
            model.current_epoch = epoch_idx
            self.timer.epoch_start()

            # training batch loop
            model.train()
            training_results = []
            for batch_idx, batch in enumerate(train_loader):
                batch = self._to_device(batch, model.device)
                # DO NOT return tensors directly, this can lead to gpu menory shortage !!
                result = model.training_step(batch, batch_idx)
                training_results.append(result)
                self.step_idx += 1
                model.current_step = self.step_idx

                # due to the potential display error of progress bar, use standard output is a wiser option.
                self.printer.batch_output(
                    'Training', epoch_idx, batch_idx, trainset_len, self.logger.last_log)
            model.on_training_end(training_results)

            # validation batch loop
            with torch.no_grad():
                model.eval()
                val_results = []
                for batch_idx, batch in enumerate(val_loader):
                    batch = self._to_device(batch, model.device)
                    # !!DO NOT return tensors directly, this can lead to gpu menory shortage !!
                    result = model.validation_step(batch, batch_idx)
                    val_results.append(result)
                    self.printer.batch_output(
                        'Validating', epoch_idx, batch_idx, valset_len, self.logger.last_log)
                model.on_validation_end(val_results)

            # epoch end
            model.on_epoch_end(training_results, val_results)

            self.logger.reduce_epoch_log(epoch_idx, self.step_idx)
            self.logger.save_log()
            model.save(self.cur_log_folder)
            self.timer.epoch_end()
            self.printer.epoch_output(
                epoch_idx, self.timer.epoch_cost, self.logger.last_log)

        # training end
        model.on_training_end()
        self.timer.training_end()
        self.printer.end_output('Traning', self.timer.total_cost)

    def test(self, model, test_loader):
        model = self.model_distribute(model)
        model.logger = self.logger
        testset_len = len(test_loader)

        # test start
        self.timer.training_start()
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nTest started\n')
        with torch.no_grad():
            model.eval()
            test_results = []
            for batch_idx, batch in enumerate(test_loader):
                batch = self._to_device(batch, model.device)
                # !!DO NOT return tensors directly, this can lead to gpu menory shortage !! use a.item() instead
                result = model.test_step(batch, batch_idx)
                test_results.append(result)
                self.printer.batch_output(
                    'Testing', 0, batch_idx, testset_len, self.logger.last_log)
            model.on_test_end(test_results)
            self.logger.save_log()
            self.printer.epoch_output(
                0, 0, self.logger.last_log)

        # test end
        self.timer.training_end()
        self.printer.end_output('Test', self.timer.total_cost)

    def _to_device(self, batch, device):
        ''' Move batch data to device automatically
            batch: should be either in the following forms -- tensor / [tensor1, tensor2,...] / [[tensor1,tensor2..],[tensor1,tensor2..],...]
            return the same form of batch with all tensor on the dedicated device
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
