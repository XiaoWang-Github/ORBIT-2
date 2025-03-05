import gptl4py as gp

# import scorep.user as sp
from contextlib import contextmanager
import torch.distributed as dist
import os
import torch


class ProfileTimer:
    def __init__(self):
        self.timers = dict()  ## active or not

    def begin(self, name, check_epoch=False):
        if name not in self.timers:
            self.timers[name] = None
        self.timers[name] = True
        gp.start(name)

    def end(self, name, check_epoch=False):
        self.timers[name] = False
        torch.cuda.synchronize()
        gp.stop(name)

    def isactive(self, name):
        return self.timers[name]
