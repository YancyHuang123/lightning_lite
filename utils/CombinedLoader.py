from ast import Tuple
from typing import List, Literal
from torch.utils.data import Dataset, DataLoader
_LITERAL_SUPPORTED_MODES = Literal["min_size",
                                   "max_size_cycle", "max_size", "sequential"]


class CombinedLoader():
    def __init__(self, iterables: List[DataLoader], mode: _LITERAL_SUPPORTED_MODES = 'min_size'):
        self.iterables = iterables
        self.mode = mode
        self.min_len = min([len(x) for x in self.iterables])
        self.max_len = max([len(x) for x in self.iterables])
        self.dataset_len = {'min_size': self.min_len,
                            'max_len': self.max_len}[self.mode] 
        self.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.batch_idx += 1
        if self.batch_idx == self.dataset_len-1:
            self.reset()
            raise StopIteration()
        return tuple([next(x) for x in self.item_list])

    def __len__(self):
        return self.dataset_len

    def reset(self):
        self.batch_idx = -1
        self.item_list = [iter(x) for x in self.iterables]