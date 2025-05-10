import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

from mpi4py import MPI
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.distributed as dist
from datetime import datetime, timedelta

try:
    import pyddstore as dds
except:
    print("DDStore loading error!!")

import re
import os


def dict2list(x, variables):
    xlist = list()
    for var in variables:
        xlist.append(x[var])
    return np.stack(xlist)


def list2dict(x, variables):
    xdict = dict()
    for i in range(len(x)):
        xdict[variables[i]] = torch.from_numpy(x[i, ...])
    return xdict


class DDStoreDataLoader(DataLoader):
    def __init__(self, ddstore, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.batch_index = 0
        # self.iterator = super().__iter__()
        self.parent = super()
        self.ddstore = ddstore

    def __iter__(self):
        self.ddstore.epoch_begin()
        is_active = True
        for batch in super().__iter__():
            is_active = False
            self.ddstore.epoch_end()
            # print("batch:", len(batch), batch[0].shape, batch[1].shape, type(batch[2]), type(batch[3]))
            yield batch
            self.ddstore.epoch_begin()
            is_active = True
        if is_active:
            self.ddstore.epoch_end()

    def collate_fn(self, batch):
        return super().collate_fn(batch)


class DistDataset(Dataset):
    """Distributed dataset class"""

    def __init__(
        self,
        dataset,
        label,
        ddp_group=None,
        comm=MPI.COMM_WORLD,
        ddstore_width=None,
    ):
        super().__init__()

        self.datasetlist = list()
        self.label = label

        self.ddp_group = ddp_group

        self.world_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        data_par_rank = dist.get_rank(group=self.ddp_group)

        color = 0
        self.comm = comm.Split(color, self.world_rank)
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        print(
            "init: rank,color,ddp_rank,ddp_size,label =",
            self.world_rank,
            color,
            self.rank,
            self.comm_size,
            label,
        )
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        print(
            "ddstore info:",
            self.world_rank,
            self.world_size,
            color,
            self.rank,
            self.ddstore_width,
        )
        self.ddstore_comm = self.comm.Split(self.rank // self.ddstore_width, self.rank)
        self.ddstore_comm_rank = self.ddstore_comm.Get_rank()
        self.ddstore_comm_size = self.ddstore_comm.Get_size()
        print(
            "ddstore MPI:",
            self.world_rank,
            self.ddstore_comm_rank,
            self.ddstore_comm_size,
        )

        ddstore_method = int(os.getenv("ORBIT_DDSTORE_METHOD", "0"))
        gpu_id = int(os.getenv("SLURM_LOCALID"))
        os.environ["FABRIC_IFACE"] = f"hsn{gpu_id//2}"
        print("DDStore method:", ddstore_method)
        print("FABRIC_IFACE:", os.environ["FABRIC_IFACE"])

        self.ddstore = dds.PyDDStore(self.ddstore_comm)

        ## register local data
        ## Assume variables and out_variables are same for all
        xlist = list()
        ylist = list()
        self.variables = None
        self.out_variables = None
        is_first = True
        for x, y, variables, out_variables in dataset:
            x_ = dict2list(x, variables)
            y_ = dict2list(y, out_variables)
            xlist.append(x_)
            ylist.append(y_)

            if is_first:
                self.variables = variables
                self.out_variables = out_variables
                is_first = False
        xarr = np.stack(xlist, dtype=np.float32)
        yarr = np.stack(ylist, dtype=np.float32)
        del xlist
        del ylist
        self.xshape = (1,) + xarr.shape[1:]
        self.yshape = (1,) + yarr.shape[1:]
        local_ns = len(xarr)
        print(
            f"[{self.rank}] DDStore: xarr.shape ",
            xarr.shape,
            xarr.size,
            f"{xarr.nbytes / 2**30:.2f} (GB)",
        )
        print(
            f"[{self.rank}] DDStore: yarr.shape ",
            yarr.shape,
            xarr.size,
            f"{yarr.nbytes / 2**30:.2f} (GB)",
        )

        self.total_ns = self.ddstore_comm.allreduce(local_ns, op=MPI.SUM)
        print("[%d] DDStore: %d %d" % (self.rank, local_ns, self.total_ns))

        self.ddstore.add(f"{self.label}-x", xarr)
        self.ddstore.add(f"{self.label}-y", yarr)
        del xarr
        del yarr
        self.ddstore_comm.Barrier()
        # print("Init done.")

    def len(self):
        return self.total_ns

    def __len__(self):
        return self.len()

    def get(self, i):
        # print ("[%d:%d] get:"%(self.world_rank, self.rank), i, self.xshape, self.yshape)

        x = np.zeros(self.xshape, dtype=np.float32)
        y = np.zeros(self.yshape, dtype=np.float32)
        self.ddstore.get(f"{self.label}-x", x, i)
        self.ddstore.get(f"{self.label}-y", y, i)
        # print ("[%d:%d] received:"%(self.world_rank, self.rank), i)

        xdict = list2dict(x[0, :], self.variables)
        ydict = list2dict(y[0, :], self.out_variables)

        return (xdict, ydict, self.variables, self.out_variables)

    def __getitem__(self, i):
        return self.get(i)
