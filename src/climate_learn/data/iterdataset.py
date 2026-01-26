# Standard library
import random
from dataclasses import dataclass

# Third party
import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass
class TileBounds:
    """Container for tile boundary coordinates.

    Attributes:
        x_start, x_end: Horizontal boundaries
        y_start, y_end: Vertical boundaries
    """

    x_start: int
    x_end: int
    y_start: int
    y_end: int


def shuffle_two_list(list1, list2):
    """Shuffle two lists in the same order to maintain pairing."""
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(list1)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1_shuf, list2_shuf


def calculate_tile_overlap(overlap):
    """Calculate tile overlap for horizontal and vertical dimensions.

    Args:
        overlap: Base overlap size

    Returns:
        tuple: (left, right, top, bottom) overlap sizes

    Note:
        Horizontal overlap is 2x vertical due to 2:1 aspect ratio of climate data (lon:lat)
    """
    if overlap % 2 == 0:
        # Even overlap: symmetric padding
        top = bottom = overlap // 2
        left = right = overlap // 2 * 2  # 2x for longitude dimension
    else:
        # Odd overlap: asymmetric padding
        left = overlap // 2 * 2  # 2x for longitude dimension
        right = (overlap // 2 + 1) * 2  # 2x for longitude dimension
        top = overlap // 2
        bottom = overlap // 2 + 1
    return left, right, top, bottom


def calculate_tile_bounds(
    tile_idx, total_tiles, dimension_size, overlap_start, overlap_end
):
    """Calculate boundaries for a single tile including overlap.

    Args:
        tile_idx: Index of current tile (0-based)
        total_tiles: Total number of tiles in this dimension
        dimension_size: Total size of dimension (width or height)
        overlap_start: Overlap size at start of tile
        overlap_end: Overlap size at end of tile

    Returns:
        tuple: (start, end) coordinates for this tile
    """
    if total_tiles == 1:
        # No tiling: use full dimension
        return 0, dimension_size

    # Base tile boundaries without overlap
    tile_size = dimension_size // total_tiles
    start = tile_size * tile_idx
    end = tile_size * (tile_idx + 1)

    # Add overlap based on tile position
    if tile_idx == 0:
        # First tile: only overlap on right
        end += overlap_start
    elif tile_idx == total_tiles - 1:
        # Last tile: only overlap on left
        start -= overlap_end
    else:
        # Middle tiles: overlap on both sides
        start -= overlap_start
        end += overlap_end

    return start, end


class NpyReader(IterableDataset):
    """Reader for NPY/NPZ files with TILES support for large climate datasets.

    This class implements the TILES algorithm to divide large climate images into
    smaller, overlapping tiles for memory-efficient processing. The overlap ensures
    smooth reconstruction when tiles are stitched back together.

    Args:
        inp_file_list: List of input NPZ files
        out_file_list: List of output NPZ files (must match inp_file_list length)
        variables: Input variable names to extract
        out_variables: Output variable names to extract
        data_par_size: Size of data parallel group
        data_par_group: Data parallel process group
        shuffle: Whether to shuffle files each epoch
        div: Number of divisions per dimension (creates div x div tiles)
        overlap: Base overlap size between tiles (actual overlap is 2x in longitude)
    """

    def __init__(
        self,
        inp_file_list,
        out_file_list,
        variables,
        out_variables,
        data_par_size: int = 1,
        data_par_group=None,
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
            self.inp_file_list = (
                self.inp_file_list * n_multiply + self.inp_file_list[:n_remain]
            )
            self.out_file_list = (
                self.out_file_list * n_multiply + self.out_file_list[:n_remain]
            )
            n_files = len(self.inp_file_list)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            rank = torch.distributed.get_rank(group=self.data_par_group)
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
                rank = torch.distributed.get_rank(group=self.data_par_group)
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * data_par_size
            per_worker = n_files // num_shards
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            path_inp = self.inp_file_list[idx]
            path_out = self.out_file_list[idx]
            print(torch.distributed.get_rank(), "NpyReader:", path_inp)

            inp_data = np.load(path_inp)
            if path_out == path_inp:
                out_data = inp_data
            else:
                out_data = np.load(path_out)

            # Get dimensions from first variable
            first_in_var = self.variables[0]
            first_out_var = self.out_variables[0]

            # Extract dimension sizes (format: [time, channels, lat, lon])
            input_width = len(inp_data[first_in_var][0, 0, 0, :])
            input_height = len(inp_data[first_in_var][0, 0, :, 0])
            output_width = len(out_data[first_out_var][0, 0, 0, :])
            output_height = len(out_data[first_out_var][0, 0, :, 0])

            # Calculate scaling factors for downscaling task
            scale_x = output_width // input_width
            scale_y = output_height // input_height

            # Calculate overlap sizes considering 2:1 aspect ratio
            left, right, top, bottom = calculate_tile_overlap(self.overlap)

            # Generate tiles in raster scan order
            for row_idx in range(self.div):
                for col_idx in range(self.div):
                    # Calculate input tile boundaries
                    x_start_in, x_end_in = calculate_tile_bounds(
                        col_idx, self.div, input_width, left, right
                    )
                    y_start_in, y_end_in = calculate_tile_bounds(
                        row_idx, self.div, input_height, top, bottom
                    )

                    # Calculate output tile boundaries (scaled by resolution ratio)
                    x_start_out, x_end_out = calculate_tile_bounds(
                        col_idx, self.div, output_width, left * scale_x, right * scale_x
                    )
                    y_start_out, y_end_out = calculate_tile_bounds(
                        row_idx,
                        self.div,
                        output_height,
                        top * scale_y,
                        bottom * scale_y,
                    )

                    # Extract tile data for all variables
                    input_tile = {
                        k: np.squeeze(
                            inp_data[k][:, :, y_start_in:y_end_in, x_start_in:x_end_in],
                            axis=1,
                        )
                        for k in self.variables
                    }
                    output_tile = {
                        k: np.squeeze(
                            out_data[k][
                                :, :, y_start_out:y_end_out, x_start_out:x_end_out
                            ],
                            axis=1,
                        )
                        for k in self.out_variables
                    }

                    yield (input_tile, output_tile, self.variables, self.out_variables)


class DirectForecast(IterableDataset):
    """Dataset for direct weather forecasting tasks.

    Args:
        dataset: Base dataset providing data samples
        src: Data source ('era5' or 'mpi-esm1-2-hr')
        pred_range: Number of timesteps to predict ahead
        history: Number of historical timesteps to use as input
        window: Timestep interval between historical frames
    """

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
    """Dataset for continuous weather forecasting with variable lead times.

    Args:
        dataset: Base dataset providing data samples
        random_lead_time: Whether to randomly vary prediction lead time
        min_pred_range: Minimum prediction range in timesteps
        max_pred_range: Maximum prediction range in timesteps
        hrs_each_step: Hours per timestep
        history: Number of historical timesteps to use as input
        window: Timestep interval between historical frames
    """

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
    """Dataset for climate downscaling tasks.

    Simply converts numpy arrays to torch tensors without temporal processing.

    Args:
        dataset: Base dataset providing low/high resolution data pairs
    """

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
    """Wrapper that applies transforms and subsampling to dataset items.

    Args:
        dataset: Base dataset to iterate over
        transforms: Dictionary of transforms for input variables
        output_transforms: Dictionary of transforms for output variables
        subsample: Subsampling factor (only yield every n-th sample)
    """

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
    """Shuffles an iterable dataset using a buffer.

    Maintains a buffer of samples and randomly yields from it to approximate
    shuffling for iterable datasets that can't be fully loaded in memory.

    Args:
        dataset: Base iterable dataset
        buffer_size: Size of shuffle buffer
    """

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
