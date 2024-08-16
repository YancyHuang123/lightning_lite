import time
from statistics import mean


class Timer():
    def __init__(self, estimate_epoch=4) -> None:
        self.experiment_start = 0
        self.experiment_end = 0
        self.epoch_start = 0
        self.epoch_end = 0

        self.experiment_cost = 0
        self.epoch_cost = 0

        # the previous k epochs that are used to estimate epoch time.
        self.estimate_epoch = estimate_epoch
        self.epoch_costs = []

    def experiment_start(self):
        self.experiment_start = time.time()

    def experiment_end(self):
        self.experiment_end = time.time()
        self.experiment_cost = self.experiment_end-self.experiment_start

    def epoch_start(self):
        self.epoch_start = time.time()

    def epoch_end(self):
        self.epoch_end = time.time()
        self.epoch_cost = self.epoch_end-self.epoch_start
        self.epoch_costs.append(self.epoch_cost)
        if len(self.epoch_costs) > self.estimate_epoch:
            self.epoch_costs.pop(0)

    def avg_epoch_cost(self):
        return mean(self.epoch_costs)
