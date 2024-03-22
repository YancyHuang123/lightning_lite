import time

class WrapperTimer():
    def __init__(self) -> None:
        self.total_cost=0
        self.epoch_cost=0

    def epoch_start(self):
        self.epoch_cost=time.time()

    def training_start(self):
        self.total_cost=time.time()

    def training_end(self):
        self.total_cost=time.time()-self.total_cost

    def epoch_end(self):
        self.epoch_cost=time.time()-self.epoch_cost

