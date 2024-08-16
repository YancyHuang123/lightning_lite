import os
import pandas as pd
import torch


# convert tensors contined in dict to scales.
def tensor_to_scale(dict: dict):
    for k, v in dict.items():
        if torch.is_tensor(v):
            dict[k] = v.item()
    return dict


# add '_epoch' to the reduced epoch data's name
def add_epoch_suffix(dict: dict, suffix: str):
    new_dict = {}
    for k, v in dict.items():
        k = k + suffix
        new_dict[k] = v
    return new_dict


class Logger():
    def __init__(self, saving_folder, log_name) -> None:
        self.saving_folder = saving_folder
        self.log = pd.DataFrame()  # the main log
        self.epoch_log = pd.DataFrame()  # log that caches current epoch data
        self.log_name = log_name
        self.last_log = {}

    def add_epoch_log(self, dict, suffix='_epoch'):
        '''add dict to epcoh log'''
        dict = tensor_to_scale(dict)
        self.last_log = dict
        # add suffix to epoch coloumns for clearifing
        dict = add_epoch_suffix(dict, suffix)
        new_row = pd.DataFrame(dict, index=[0])
        self.epoch_log = pd.concat(
            [self.epoch_log, new_row], ignore_index=True)

    def reduce_epoch_log(self, epoch=None, step=None):
        '''reduce epoch log (self.epoch_log) and log it to main log (self.log)'''
        # the mean value of each column
        mu = self.epoch_log.mean(axis=0, skipna=True).to_frame().T
        self.last_log = mu.iloc[0].to_dict()
        mu['epoch'] = epoch  # add epoch_idx and step_idx info
        mu['step'] = step

        # concatanate mean values to log
        self.log = pd.concat([self.log, mu], ignore_index=True)

        self.epoch_log = pd.DataFrame()  # clear epoch_log

    def add_log(self, dict, epoch=None, step=None):  # directly add to main log
        dict = tensor_to_scale(dict)
        self.last_log = dict
        dict['epoch'] = epoch
        dict['step'] = step
        new_row = pd.DataFrame(dict, index=[0])
        self.log = pd.concat([self.log, new_row], ignore_index=True)

    def save_log(self):
        self.log.to_csv(f'{self.saving_folder}/{self.log_name}', index=False)
