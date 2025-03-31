# Standard library
import copy
import glob
import os
from typing import Dict, Optional, OrderedDict

# Third party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

# Local application
from .iterdataset import (
    NpyReader,
    DirectForecast,
    ContinuousForecast,
    Downscale,
    IndividualDataIter,
    ShuffleIterableDataset,
)
from .processing.era5_constants import PRECIP_VARIABLES
from .precipmodule import LogTransform

from climate_learn.dist.distdataset import *
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


class IterDataModule(torch.nn.Module):
    """ClimateLearn's iter data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        task,
        inp_root_dir,
        out_root_dir,
        in_vars,
        out_vars,
        src=None,
        history=1,
        window=6,
        pred_range=6,
        random_lead_time=True,
        max_pred_range=120,
        hrs_each_step=1,
        subsample=1,
        buffer_size=10000,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()
        self.task = task
        self.inp_root_dir = inp_root_dir
        self.out_root_dir = out_root_dir
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.subsample = subsample
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.ddp_group = dist.distributed_c10d._get_default_group()

        if task in ("direct-forecasting", "iterative-forecasting"):
            self.dataset_caller = DirectForecast
            self.dataset_arg = {
                "src": src,
                "pred_range": pred_range,
                "history": history,
                "window": window,
            }
            self.collate_fn = collate_fn
        elif task == "continuous-forecasting":
            self.dataset_caller = ContinuousForecast
            self.dataset_arg = {
                "random_lead_time": random_lead_time,
                "min_pred_range": pred_range,
                "max_pred_range": max_pred_range,
                "hrs_each_step": hrs_each_step,
                "history": history,
                "window": window,
            }
            self.collate_fn = collate_fn_continuous
        elif task == "downscaling":
            self.dataset_caller = Downscale
            self.dataset_arg = {}
            self.collate_fn = collate_fn

        self.inp_lister_train = sorted(
            glob.glob(os.path.join(inp_root_dir, "train", "*.npz"))
        )
        self.out_lister_train = sorted(
            glob.glob(os.path.join(out_root_dir, "train", "*.npz"))
        )
        self.inp_lister_val = sorted(
            glob.glob(os.path.join(inp_root_dir, "val", "*.npz"))
        )
        self.out_lister_val = sorted(
            glob.glob(os.path.join(out_root_dir, "val", "*.npz"))
        )
        self.inp_lister_test = sorted(
            glob.glob(os.path.join(inp_root_dir, "test", "*.npz"))
        )
        self.out_lister_test = sorted(
            glob.glob(os.path.join(out_root_dir, "test", "*.npz"))
        )

        self.transforms = self.get_normalize(inp_root_dir, in_vars)
        self.output_transforms = self.get_normalize(out_root_dir, out_vars)
        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.out_root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.out_root_dir, "lon.npy"))
        return lat, lon

    def get_data_variables(self):
        out_vars = copy.deepcopy(self.out_vars)
        if "2m_temperature_extreme_mask" in out_vars:
            out_vars.remove("2m_temperature_extreme_mask")
        return self.in_vars, out_vars

    def get_data_dims(self):
        in_lat = len(np.load(os.path.join(self.inp_root_dir, "lat.npy")))
        in_lon = len(np.load(os.path.join(self.inp_root_dir, "lon.npy")))
        out_lat = len(np.load(os.path.join(self.out_root_dir, "lat.npy")))
        out_lon = len(np.load(os.path.join(self.out_root_dir, "lon.npy")))

        forecasting_tasks = [
            "direct-forecasting",
            "iterative-forecasting",
            "continuous-forecasting",
        ]
        if self.task in forecasting_tasks:
            in_size = torch.Size(
                [
                    self.batch_size,
                    self.history,
                    len(self.in_vars),
                    out_lat,
                    out_lon,
                ]
            )
        elif self.task == "downscaling":
            in_size = torch.Size(
                [self.batch_size, len(self.in_vars), in_lat, in_lon]
            )
        ##TODO: change out size
        out_vars = copy.deepcopy(self.out_vars)
        if "2m_temperature_extreme_mask" in out_vars:
            out_vars.remove("2m_temperature_extreme_mask")
        out_size = torch.Size([self.batch_size, len(out_vars), out_lat, out_lon])
        return in_size, out_size

    def get_normalize(self, root_dir, variables):
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normed = OrderedDict()
        for var in variables:
            if var in PRECIP_VARIABLES:
                normed[var] = LogTransform(m2mm=True, LOG1P=True, thres_mm_per_day=0.25) 
            else:
                normed[var] = transforms.Normalize(normalize_mean[var][0], normalize_std[var][0])
        return normed
 
    def get_out_transforms(self):
        out_transforms = {}
        for key in self.output_transforms.keys():
            if key == "2m_temperature_extreme_mask":
                continue
            out_transforms[key] = self.output_transforms[key]
        return out_transforms

    def get_climatology(self, split="val"):
        path = os.path.join(self.out_root_dir, split, "climatology.npz")
        clim_dict = np.load(path)
        new_clim_dict = {}
        for var in self.out_vars:
            if var == "2m_temperature_extreme_mask":
                continue
            new_clim_dict[var] = torch.from_numpy(
                np.squeeze(clim_dict[var].astype(np.float32), axis=0)
            )
        return new_clim_dict

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        use_ddstore = int(os.environ.get("ORBIT_USE_DDSTORE", 0))
        print("use_ddstore is :", use_ddstore, flush=True)
        if use_ddstore:
            self.data_train = IndividualDataIter(
                self.dataset_caller(
                    NpyReader(
                        inp_file_list=self.inp_lister_train,
                        out_file_list=self.out_lister_train,
                        variables=self.in_vars,
                        out_variables=self.out_vars,
                        shuffle=True,
                    ),
                    **self.dataset_arg,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
                subsample=self.subsample,
            )

            self.data_val = IndividualDataIter(
                self.dataset_caller(
                    NpyReader(
                        inp_file_list=self.inp_lister_val,
                        out_file_list=self.out_lister_val,
                        variables=self.in_vars,
                        out_variables=self.out_vars,
                        shuffle=False,
                    ),
                    **self.dataset_arg,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
                subsample=self.subsample,
            )

            self.data_test = IndividualDataIter(
                self.dataset_caller(
                    NpyReader(
                        inp_file_list=self.inp_lister_test,
                        out_file_list=self.out_lister_test,
                        variables=self.in_vars,
                        out_variables=self.out_vars,
                        shuffle=False,
                    ),
                    **self.dataset_arg,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
                subsample=self.subsample,
            )
            return

        if stage != "test":
            if not self.data_train and not self.data_val and not self.data_test:
                self.data_train = ShuffleIterableDataset(
                    IndividualDataIter(
                        self.dataset_caller(
                            NpyReader(
                                inp_file_list=self.inp_lister_train,
                                out_file_list=self.out_lister_train,
                                variables=self.in_vars,
                                out_variables=self.out_vars,
                                shuffle=True,
                            ),
                            **self.dataset_arg,
                        ),
                        transforms=self.transforms,
                        output_transforms=self.output_transforms,
                        subsample=self.subsample,
                    ),
                    buffer_size=self.buffer_size,
                )

                self.data_val = IndividualDataIter(
                    self.dataset_caller(
                        NpyReader(
                            inp_file_list=self.inp_lister_val,
                            out_file_list=self.out_lister_val,
                            variables=self.in_vars,
                            out_variables=self.out_vars,
                            shuffle=False,
                        ),
                        **self.dataset_arg,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    subsample=self.subsample,
                )

                self.data_test = IndividualDataIter(
                    self.dataset_caller(
                        NpyReader(
                            inp_file_list=self.inp_lister_test,
                            out_file_list=self.out_lister_test,
                            variables=self.in_vars,
                            out_variables=self.out_vars,
                            shuffle=False,
                        ),
                        **self.dataset_arg,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    subsample=self.subsample,
                )
        else:
            self.data_test = IndividualDataIter(
                self.dataset_caller(
                    NpyReader(
                        inp_file_list=self.inp_lister_test,
                        out_file_list=self.out_lister_test,
                        variables=self.in_vars,
                        out_variables=self.out_vars,
                        shuffle=False,
                    ),
                    **self.dataset_arg,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
                subsample=self.subsample,
            )

    def train_dataloader(self):
        use_ddstore = int(os.environ.get("ORBIT_USE_DDSTORE", 0))
        # print("use_ddstore is :", use_ddstore, flush=True)

        if use_ddstore:
            ## assume: a GPU is mapped by the local rank
            gpu_id = int(os.getenv("SLURM_LOCALID", "0"))
            os.environ["FABRIC_IFACE"] = f"hsn{gpu_id//2}"
            print("FABRIC_IFACE:", os.environ["FABRIC_IFACE"])

            ddp_group_size = dist.get_world_size(group=self.ddp_group)
            ddp_group_rank = dist.get_rank(group=self.ddp_group)

            trainset = DistDataset(
                self.data_train,
                "trainset",
                ddp_group = self.ddp_group,
                )

            ddstore_sampler = os.getenv("ORBIT_DDSTORE_SAMPLER", "local") ## 0: global 1:local
            if ddp_group_rank == 0:
                print("DDStore sampler:", ddstore_sampler)
            if ddstore_sampler == "global":
                sampler = DistributedSampler(trainset, num_replicas=ddp_group_size, rank=ddp_group_rank, shuffle=True)
            elif ddstore_sampler == "local":
                rx = list(nsplit(range(len(trainset)), ddp_group_size))[ddp_group_rank]
                sampler = SubsetRandomSampler(rx)

            ddstore_method = int(os.getenv("ORBIT_DDSTORE_METHOD", "0"))
            if ddstore_method == 0:
                train_loader = DDStoreDataLoader(
                    trainset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=True,
                    sampler=sampler,
                    collate_fn=collate_fn,
                )
            else:
                ## multi-thread
                train_loader = HydraDataLoader(
                    trainset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=True,
                    num_workers=self.num_workers,
                    sampler=sampler,
                    collate_fn=collate_fn,
                )

            return train_loader

        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )


def collate_fn(batch):
    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        t = torch.stack(tuple(t.values()))
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    has_extreme_mask = False
    for key in batch[0][1]:
        if key == "2m_temperature_extreme_mask":
            has_extreme_mask = True
    if not has_extreme_mask:
        out = torch.stack(
            [handle_dict_features(batch[i][1]) for i in range(len(batch))]
        )
        variables = list(batch[0][0].keys())
        out_variables = list(batch[0][1].keys())
        return inp, out, variables, out_variables
    out = []
    mask = []
    for i in range(len(batch)):
        out_dict = {}
        mask_dict = {}
        for key in batch[i][1]:
            if key == "2m_temperature_extreme_mask":
                mask_dict[key] = batch[i][1][key]
            else:
                out_dict[key] = batch[i][1][key]
        out.append(handle_dict_features(out_dict))
        if mask_dict != {}:
            mask.append(handle_dict_features(mask_dict))
    out = torch.stack(out)
    if mask != []:
        mask = torch.stack(mask)
    variables = list(batch[0][0].keys())
    out_variables = list(out_dict.keys())
    return inp, out, mask, variables, out_variables


def collate_fn_continuous(batch):
    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        t = torch.stack(tuple(t.values()))
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    b, t, _, h, w = inp.shape
    lead_times = lead_times.reshape(b, 1, 1, 1, 1).repeat(1, t, 1, h, w)
    inp = torch.cat((inp, lead_times), dim=2)
    variables = list(batch[0][0].keys())
    out_variables = list(batch[0][1].keys())
    return inp, out, variables, out_variables
