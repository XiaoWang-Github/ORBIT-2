# Standard library
import random

# Third party
import numpy as np
import torch
from torch.utils.data import IterableDataset


def shuffle_two_list(list1, list2):
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(list1)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1_shuf, list2_shuf


class NpyReader(IterableDataset):
    def __init__(
        self,
        inp_file_list,
        out_file_list,
        variables,
        out_variables,
        data_par_size: int =1,
        data_par_group = None,
        shuffle=False,
        div=1,
        overlap=4,
    ):
        super().__init__()
        assert len(inp_file_list) == len(out_file_list)
        self.inp_file_list = [f for f in inp_file_list if "climatology" not in f]
        self.out_file_list = [f for f in out_file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle
        self.data_par_size = data_par_size
        self.data_par_group = data_par_group
        self.div = div
        self.overlap = overlap

    def __iter__(self):
        if self.shuffle:
            self.inp_file_list, self.out_file_list = shuffle_two_list(
                self.inp_file_list, self.out_file_list
            )

        n_files = len(self.inp_file_list)

        ## Wrap-around filelist if files < processes.
        data_par_size = self.data_par_size if torch.distributed.is_initialized() else 1
        worker_info = torch.utils.data.get_worker_info()
        num_workers_per_ddp = worker_info.num_workers if worker_info is not None else 1
        total_num_workers = num_workers_per_ddp * self.data_par_size


        if n_files < total_num_workers:
            n_multiply = total_num_workers // n_files
            n_remain = total_num_workers - n_files * n_multiply
            self.inp_file_list = self.inp_file_list * n_multiply + self.inp_file_list[:n_remain]
            self.out_file_list = self.out_file_list * n_multiply + self.out_file_list[:n_remain]
            n_files = len(self.inp_file_list)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            rank = torch.distributed.get_rank(group = self.data_par_group)
            num_workers_per_ddp = 1
            num_shards = num_workers_per_ddp * data_par_size
            per_worker = n_files // num_shards
            worker_id = rank * num_workers_per_ddp
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                data_par_size = 1
            else:
                rank = torch.distributed.get_rank(group = self.data_par_group)
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * data_par_size
            per_worker = n_files // num_shards
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        div_size = self.div ** 2

        for idx in range(iter_start, iter_end):
            path_inp = self.inp_file_list[idx]
            path_out = self.out_file_list[idx]
            print(torch.distributed.get_rank(), "NpyReader:", path_inp)

            inp_data = np.load(path_inp)
            if path_out == path_inp:
                out_data = inp_data
            else:
                out_data = np.load(path_out)

            k0 = self.variables[0]
            k1 = self.out_variables[0]
            xinp = len( inp_data[k0][0,0,0,:] )
            yinp = len( inp_data[k0][0,0,:,0] )
            xout = len( out_data[k1][0,0,0,:] )
            yout = len( out_data[k1][0,0,:,0] )
            hmul = xout // xinp
            vmul = yout // yinp

            if self.overlap % 2 == 0:
                left = right = self.overlap // 2 * 2
                top = bottom = self.overlap // 2
            else:
                left = self.overlap // 2 * 2
                right = ( self.overlap // 2 + 1 ) * 2
                top = self.overlap // 2
                bottom = self.overlap // 2 + 1
            #hoverlap = self.overlap * 2
            #voverlap = self.overlap

            for vindex in range(self.div):
                for hindex in range(self.div):

                    if self.div == 1:
                        xi1 = 0
                        xi2 = xinp
                        xo1 = 0
                        xo2 = xout
                    else:
                        xi1 = xinp // self.div * hindex
                        xi2 = xinp // self.div * (hindex+1)
                        xo1 = xout // self.div * hindex
                        xo2 = xout // self.div * (hindex+1)
                        if hindex == 0:
                            xi2 += left
                            xo2 += left * hmul
                        else:
                            xi1 -= left
                            xo1 -= left * hmul
                        if hindex == self.div - 1:
                            xi1 -= right
                            xo1 -= right * hmul
                        else:
                            xi2 += right
                            xo2 += right * hmul

                    if self.div == 1:
                        yi1 = 0
                        yi2 = yinp
                        yo1 = 0
                        yo2 = yout
                    else:
                        yi1 = yinp // self.div * vindex
                        yi2 = yinp // self.div * (vindex+1)
                        yo1 = yout // self.div * vindex
                        yo2 = yout // self.div * (vindex+1)
                        if vindex == 0:
                            yi2 += top
                            yo2 += top * vmul
                        else:
                            yi1 -= top
                            yo1 -= top * vmul
                        if vindex == self.div - 1:
                            yi1 -= bottom
                            yo1 -= bottom * vmul
                        else:
                            yi2 += bottom
                            yo2 += bottom * vmul

                    yield (
                        {k: np.squeeze(inp_data[k][:,:,yi1:yi2,xi1:xi2], axis=1) for k in self.variables},
                        {k: np.squeeze(out_data[k][:,:,yo1:yo2,xo1:xo2], axis=1) for k in self.out_variables},
                        self.variables,
                        self.out_variables
                    )

        """
        for idx in range(iter_start, iter_end):
            path_inp = self.inp_file_list[idx]
            path_out = self.out_file_list[idx]
            print("world_rank",torch.distributed.get_rank(),"data_par_group rank",rank, "NpyReader:", path_inp)
            inp = np.load(path_inp)
            if path_out == path_inp:
                out = inp
            else:
                out = np.load(path_out)
            yield {k: np.squeeze(inp[k], axis=1) for k in self.variables}, {
                k: np.squeeze(out[k], axis=1) for k in self.out_variables
            }, self.variables, self.out_variables
        """

class DirectForecast(IterableDataset):
    def __init__(self, dataset, src, pred_range=6, history=3, window=6):
        super().__init__()
        self.dataset = dataset
        self.history = history
        if src == "era5":
            self.pred_range = pred_range
            self.window = window
        elif src == "mpi-esm1-2-hr":
            assert pred_range % 6 == 0
            assert window % 6 == 0
            self.pred_range = pred_range // 6
            self.window = window // 6

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                .unsqueeze(0)
                .repeat_interleave(self.history, dim=0)
                for k in inp_data.keys()
            }
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            for key in inp_data.keys():
                for t in range(self.history):
                    inp_data[key][t] = inp_data[key][t].roll(-t * self.window, dims=0)

            last_idx = -((self.history - 1) * self.window + self.pred_range)

            inp_data = {
                k: inp_data[k][:, :last_idx].transpose(0, 1)
                for k in inp_data.keys()  # N, T, H, W
            }

            inp_data_len = inp_data[variables[0]].size(0)

            predict_ranges = torch.ones(inp_data_len).to(torch.long) * self.pred_range
            output_ids = (
                torch.arange(inp_data_len)
                + (self.history - 1) * self.window
                + predict_ranges
            )
            out_data = {k: out_data[k][output_ids] for k in out_data.keys()}
            yield inp_data, out_data, variables, out_variables


class ContinuousForecast(IterableDataset):
    def __init__(
        self,
        dataset,
        random_lead_time=True,
        min_pred_range=6,
        max_pred_range=120,
        hrs_each_step=1,
        history=3,
        window=6,
    ):
        super().__init__()
        if not random_lead_time:
            assert min_pred_range == max_pred_range
        self.dataset = dataset
        self.random_lead_time = random_lead_time
        self.min_pred_range = min_pred_range
        self.max_pred_range = max_pred_range
        self.hrs_each_step = hrs_each_step
        self.history = history
        self.window = window

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                .unsqueeze(0)
                .repeat_interleave(self.history, dim=0)
                for k in inp_data.keys()
            }
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            for key in inp_data.keys():
                for t in range(self.history):
                    inp_data[key][t] = inp_data[key][t].roll(-t * self.window, dims=0)

            last_idx = -((self.history - 1) * self.window + self.max_pred_range)

            inp_data = {
                k: inp_data[k][:, :last_idx].transpose(0, 1)
                for k in inp_data.keys()  # N, T, H, W
            }

            inp_data_len = inp_data[variables[0]].size(0)
            dtype = inp_data[variables[0]].dtype

            if self.random_lead_time:
                predict_ranges = torch.randint(
                    low=self.min_pred_range,
                    high=self.max_pred_range + 1,
                    size=(inp_data_len,),
                )
            else:
                predict_ranges = (
                    torch.ones(inp_data_len).to(torch.long) * self.max_pred_range
                )
            lead_times = self.hrs_each_step * predict_ranges / 100
            lead_times = lead_times.to(dtype)
            output_ids = (
                torch.arange(inp_data_len)
                + (self.history - 1) * self.window
                + predict_ranges
            )

            out_data = {k: out_data[k][output_ids] for k in out_data.keys()}
            yield inp_data, out_data, lead_times, variables, out_variables


class Downscale(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                for k in inp_data.keys()
            }
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            yield inp_data, out_data, variables, out_variables


class IndividualDataIter(IterableDataset):
    def __init__(
        self,
        dataset,
        transforms,
        output_transforms,
        subsample=6,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.subsample = subsample

    def __iter__(self):
        for sample in self.dataset:
            if isinstance(self.dataset, (DirectForecast, Downscale)):
                inp, out, variables, out_variables = sample
            elif isinstance(self.dataset, ContinuousForecast):
                inp, out, lead_times, variables, out_variables = sample
            inp_shapes = set([inp[k].shape[0] for k in inp.keys()])
            out_shapes = set([out[k].shape[0] for k in out.keys()])
            assert len(inp_shapes) == 1
            assert len(out_shapes) == 1
            inp_len = next(iter(inp_shapes))
            out_len = next(iter(out_shapes))
            assert inp_len == out_len
            for i in range(0, inp_len, self.subsample):
                x = {k: inp[k][i] for k in inp.keys()}
                y = {k: out[k][i] for k in out.keys()}
                if self.transforms is not None:
                    if isinstance(self.dataset, (DirectForecast, ContinuousForecast)):
                        x = {
                            k: self.transforms[k](x[k].unsqueeze(1)).squeeze(1)
                            for k in x.keys()
                        }
                    elif isinstance(self.dataset, Downscale):
                        x = {
                            k: self.transforms[k](x[k].unsqueeze(0)).squeeze(0)
                            for k in x.keys()
                        }
                    else:
                        raise RuntimeError(f"Not supported task.")
                if self.output_transforms is not None:
                    y = {
                        k: self.output_transforms[k](y[k].unsqueeze(0)).squeeze(0)
                        for k in y.keys()
                    }
                if isinstance(self.dataset, (DirectForecast, Downscale)):
                    result = x, y, variables, out_variables
                elif isinstance(self.dataset, ContinuousForecast):
                    result = x, y, lead_times[i], variables, out_variables
                yield result


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()
