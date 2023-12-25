import pandas as pd
import torch


class WrapperPrinter():
    def __init__(self, output_interval,max_epochs) -> None:
        self.output_interval=output_interval
        self.max_epochs=max_epochs
        self.last_log = {}

    def batch_output(self,phase:str,epoch_idx,batch_idx,loader_len,last_log):
        if batch_idx % self.output_interval == 0:
            print(
                f'>>>Epoch[{epoch_idx}] {phase} batch:{batch_idx}/{loader_len} {last_log}')

    def epoch_output(self,epoch_idx,epoch_elapse,last_log):
        print(
                f'Epoch[{epoch_idx}/{self.max_epochs-1}] ETA:{epoch_elapse/60.*(self.max_epochs-epoch_idx-1):.02f}min({epoch_elapse/60:.02f}min/epoch) {last_log}')
        
    def end_output(self,phase,consumption):
        print(
            f'{phase} completed. Time consumption:{consumption/60.:.02f}min')
