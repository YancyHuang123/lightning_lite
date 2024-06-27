from datetime import datetime
import os
import torch
import torch.nn as nn
from .LiteModule import LiteModule


def to_device(batch, device):
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
            raise Exception("data type can't be moved to device")
    return tuple(items) if len(items) != 1 else items[0]


def model_distribute(model: LiteModule, accelerator, distribution) -> LiteModule:
    '''move and distribute model to device(s)'''
    if accelerator == 'gpu':
        model.device = 'cuda'
        for key in model._modules:
            value = getattr(model, key)
            if key not in model.cuda_ignore:
                if key not in model.distribution_ignore:
                    if distribution == False:
                        value = value.to('cuda')
                    else:
                        value = nn.DataParallel(value).to('cuda')
                else:
                    value = value.to('cuda')
                setattr(model, key, value)
    return model


#def create_saving_folder():
#    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#    folder = self.log_folder
#    os.makedirs(f'{folder}', exist_ok=True)
#    os.mkdir(f"{folder}/{time}")
#    self.cur_log_folder = f"{folder}/{time}"
