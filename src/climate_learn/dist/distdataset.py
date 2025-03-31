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

from torch.utils.data.dataloader import _DatasetKind
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import queue
import socket
import logging


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

## Credit: HydraGNN
class HydraDataLoader(DataLoader):
    """
    A custom data loader with multi-threading on a HPC system.
    This is to overcome a few problems (affinity, hanging, crashing, etc)
    with Pytorch's multi-threaded DataLoader on Summit and Perlmutter.
    (2022/08) jyc: This is a work-in-progress version. Performance is not verified.
    """

    def __init__(self, dataset, **kwargs):
        super(HydraDataLoader, self).__init__(dataset, **kwargs)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self.dataset,
            self._auto_collation,
            self.collate_fn,
            self.drop_last,
        )

        ## List of threads job (futures)
        self.fs = queue.Queue()

        logging.debug("num_workers:", self.num_workers)
        logging.debug("len:", len(self._index_sampler))

    @staticmethod
    def worker_init(counter):
        core_width = 2
        if os.getenv("HYDRAGNN_AFFINITY_WIDTH") is not None:
            core_width = int(os.environ["HYDRAGNN_AFFINITY_WIDTH"])

        core_offset = 0
        if os.getenv("HYDRAGNN_AFFINITY_OFFSET") is not None:
            core_offset = int(os.environ["HYDRAGNN_AFFINITY_OFFSET"])

        with counter.get_lock():
            wid = counter.value
            counter.value += 1

        affinity = None
        if hasattr(os, "sched_getaffinity"):
            affinity_check = os.getenv("HYDRAGNN_AFFINITY")
            if affinity_check == "OMP":
                affinity = parse_omp_places(os.getenv("OMP_PLACES"))
            else:
                affinity = list(os.sched_getaffinity(0))

            affinity_mask = set(
                affinity[
                    core_width * wid
                    + core_offset : core_width * (wid + 1)
                    + core_offset
                ]
            )
            os.sched_setaffinity(0, affinity_mask)
            affinity = os.sched_getaffinity(0)

        hostname = socket.gethostname()
        logging.debug(
            f"Worker: pid={os.getpid()} hostname={hostname} ID={wid} affinity={affinity}"
        )
        return 0

    @staticmethod
    def fetch(dataset, ibatch, index, pin_memory=False):
        batch = [dataset[i] for i in index]
        # hostname = socket.gethostname()
        # log (f"Worker done: pid={os.getpid()} hostname={hostname} ibatch={ibatch}")
        # data = Batch.from_data_list(batch) ## for pytorch geometric
        if pin_memory:
            data = torch.utils.data._utils.pin_memory.pin_memory(data)
        return (ibatch, batch)

    def __iter__(self):
        logging.debug("Iterator reset")
        ## Check previous futures
        if self.fs.qsize() > 0:
            logging.debug("Clearn previous futures:", self.fs.qsize())
            for future in iter(self.fs.get, None):
                future.cancel()

        ## Resetting
        self._num_yielded = 0
        self._sampler_iter = iter(self._index_sampler)
        self.fs_iter = iter(self.fs.get, None)
        counter = mp.Value("i", 0)
        executor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            initializer=self.worker_init,
            initargs=(counter,),
        )
        for i in range(len(self._index_sampler)):
            index = next(self._sampler_iter)
            future = executor.submit(
                self.fetch,
                self.dataset,
                i,
                index,
                pin_memory=self.pin_memory,
            )
            self.fs.put(future)
        self.fs.put(None)
        # log ("Submit all done.")
        return self

    def __next__(self):
        # log ("Getting next", self._num_yielded)
        future = next(self.fs_iter)
        ibatch, data = future.result()
        # log (f"Future done: ibatch={ibatch}", data.num_graphs)
        self._num_yielded += 1
        if self.collate_fn is not None:
            data = self.collate_fn(data)
        return data

    def clean(self):
        if self.fs.qsize() > 0:
            logging.debug("Clearn previous futures:", self.fs.qsize())
            for future in iter(self.fs.get, None):
                future.cancel()


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

        self.ddstore = dds.PyDDStore(self.ddstore_comm, method=ddstore_method)

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
