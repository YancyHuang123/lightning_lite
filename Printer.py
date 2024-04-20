import pandas as pd
import torch
import math

class Printer():
    def __init__(self, output_interval,max_epochs) -> None:
        self.output_interval=output_interval
        self.max_epochs=max_epochs
        self.last_log = {}

    def format_control(self,log):
        for key, value in log.items():
            log[key]=math.trunc(value * 1000) / 1000 if isinstance(value,float) else value
        return log

    #def output_log(self,log):
    #    s=''
    #    i=0
    #    for key, value in log.items():
    #        if i!=0:
    #            s+=', '
    #        s+=f'{key}: {value if isinstance(value,float) else value}'
    #        i+=1
    #    return s
    
    def batch_output(self,phase:str,epoch_idx,batch_idx,loader_len,last_log):
        log=self.format_control(last_log)
        if batch_idx % self.output_interval == 0:
            print(
                f"{phase} Epoch[{epoch_idx}] batch:{batch_idx}/{loader_len} {log}")

    def epoch_output(self,epoch_idx,epoch_elapse,last_log):
        log=self.format_control(last_log)
        print(
                f'Epoch[{epoch_idx}/{self.max_epochs-1}] ETA:{epoch_elapse/60.*(self.max_epochs-epoch_idx-1):.02f}min({epoch_elapse/60:.02f}min/epoch) {log}')
        
    def end_output(self,phase,consumption):
        print(
            f'\n{phase} completed. Time consumption:{consumption/60.:.02f}min\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
