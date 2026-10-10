"""Train the minimal Sparse-Reslim deterministic forecasting example."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from .config import ConfigError, load_forecast_config
    from .distributed import (
        DistributedContext,
        cleanup_distributed,
        initialize_distributed,
        unwrap_model,
        wrap_model,
    )
    from .model import SparseReslim
    from .tiling import build_tile_specs, extract_tile, stitch_tiles
    from .utils import seed_everything
except ImportError:  # Support `python examples/.../train.py` from the repo root.
    from config import ConfigError, load_forecast_config
    from distributed import (
        DistributedContext,
        cleanup_distributed,
        initialize_distributed,
        unwrap_model,
        wrap_model,
    )
    from model import SparseReslim
    from tiling import build_tile_specs, extract_tile, stitch_tiles
    from utils import seed_everything


def _as_time_lat_lon(array: np.ndarray, variable: str) -> np.ndarray:
    """Normalize common ORBIT NPZ layouts to [time, latitude, longitude]."""
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(
            f"{variable!r} must have shape [T,H,W] or [T,1,H,W]; got {array.shape}"
        )
    return array


def _load_normalization(root: Path, variables: tuple[str, ...]):
    means_path = root / "normalize_mean.npz"
    stds_path = root / "normalize_std.npz"
    if not means_path.exists() or not stds_path.exists():
        raise FileNotFoundError(
            "Expected normalize_mean.npz and normalize_std.npz in the ERA5 root"
        )
    with np.load(means_path) as mean_file, np.load(stds_path) as std_file:
        means = {
            name: float(np.asarray(mean_file[name]).reshape(-1)[0])
            for name in variables
        }
        stds = {
            name: float(np.asarray(std_file[name]).reshape(-1)[0])
            for name in variables
        }
    if any(value == 0 for value in stds.values()):
        raise ValueError("normalization standard deviations must be non-zero")
    return means, stds


class ERA5ForecastDataset(IterableDataset):
    """Stream direct-forecast pairs from ORBIT-style yearly NPZ files."""

    def __init__(
        self,
        root: Path,
        split: str,
        input_variables: tuple[str, ...],
        output_variables: tuple[str, ...],
        *,
        history: int,
        window: int,
        pred_range: int,
        shuffle_files: bool,
        do_tiling: bool = False,
        tile_div: int = 1,
        tile_overlap: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.files = sorted((root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found in {root / split}")
        self.input_variables = input_variables
        self.output_variables = output_variables
        all_variables = tuple(dict.fromkeys(input_variables + output_variables))
        self.means, self.stds = _load_normalization(root, all_variables)
        self.history = history
        self.window = window
        self.pred_range = pred_range
        self.shuffle_files = shuffle_files
        self.do_tiling = do_tiling
        self.tile_div = tile_div
        self.tile_overlap = tile_overlap
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        files = list(self.files)
        if self.world_size > 1 and len(files) >= self.world_size:
            usable_files = len(files) // self.world_size * self.world_size
            files = files[:usable_files][self.rank :: self.world_size]
        if self.shuffle_files:
            random.shuffle(files)
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        for path in files:
            with np.load(path) as data:
                required = set(self.input_variables + self.output_variables)
                missing = required - set(data.files)
                if missing:
                    raise KeyError(f"{path} is missing variables: {sorted(missing)}")
                arrays = {
                    name: _as_time_lat_lon(data[name], name)
                    for name in required
                }

            total_steps = min(array.shape[0] for array in arrays.values())
            first_input = (self.history - 1) * self.window
            last_input = total_steps - self.pred_range
            for time_index in range(first_input, last_input):
                history_indices = [
                    time_index - offset * self.window
                    for offset in reversed(range(self.history))
                ]
                inputs = np.stack(
                    [
                        np.stack(
                            [
                                (arrays[name][index] - self.means[name])
                                / self.stds[name]
                                for name in self.input_variables
                            ],
                            axis=0,
                        )
                        for index in history_indices
                    ],
                    axis=0,
                )
                target_index = time_index + self.pred_range
                targets = np.stack(
                    [
                        (arrays[name][target_index] - self.means[name])
                        / self.stds[name]
                        for name in self.output_variables
                    ],
                    axis=0,
                )
                input_tensor = torch.from_numpy(inputs.astype(np.float32))
                target_tensor = torch.from_numpy(targets.astype(np.float32))
                if not self.do_tiling:
                    yield input_tensor, target_tensor
                    continue

                specs = build_tile_specs(
                    tuple(input_tensor.shape[-2:]),
                    self.tile_div,
                    self.tile_overlap,
                )
                for spec in specs:
                    yield extract_tile(input_tensor, spec), extract_tile(
                        target_tensor, spec
                    )


def _infer_image_size(root: Path, variable: str) -> tuple[int, int]:
    candidates = sorted((root / "train").glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No training NPZ files found in {root / 'train'}")
    with np.load(candidates[0]) as data:
        sample = _as_time_lat_lon(data[variable], variable)
    return int(sample.shape[-2]), int(sample.shape[-1])


def _build_model(args, img_size: tuple[int, int]) -> SparseReslim:
    return SparseReslim(
        args.input_vars,
        args.output_vars,
        img_size,
        history=args.history,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        keep_ratio=args.keep_ratio,
        num_dense_early=args.num_dense_early,
        num_sparse_middle=args.num_sparse_middle,
        token_dropping=args.token_dropping,
        compression_enabled=args.compression["enabled"],
        compress_ratio=args.compression["compress_ratio"],
        dropout=args.dropout,
    )


def _resolve_model_image_size(
    image_size: tuple[int, int], tiling
) -> tuple[int, int]:
    if not tiling["do_tiling"]:
        return image_size
    specs = build_tile_specs(image_size, tiling["div"], tiling["overlap"])
    return specs[0].shape


def _select_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "trainer.accelerator is 'gpu', but PyTorch cannot access a GPU"
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context(device: torch.device, data_type: str):
    if device.type == "cuda" and data_type == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _run_epoch(
    model,
    loader,
    device: torch.device,
    data_type: str,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, data_type):
                forecast = model(inputs)
                loss = F.mse_loss(forecast, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Encountered non-finite MSE loss: {loss.item()}")
            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(inputs.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size

    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor(
            [total_loss, total_samples], dtype=torch.float64, device=device
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss = float(totals[0].item())
        total_samples = int(totals[1].item())
    if total_samples == 0:
        raise RuntimeError("The data loader did not produce any forecast samples")
    return total_loss / total_samples


def _load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    context: DistributedContext | None = None,
):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if context is not None and context.uses_fsdp:
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, load_policy
        ):
            model.load_state_dict(state_dict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer_state = FSDP.optim_state_dict_to_load(
                model, optimizer, checkpoint["optimizer_state_dict"]
            )
            optimizer.load_state_dict(optimizer_state)
    else:
        unwrap_model(model).load_state_dict(state_dict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mse: float,
    epochs_without_improvement: int,
    context: DistributedContext,
) -> None:
    if context.is_main:
        path.parent.mkdir(parents=True, exist_ok=True)
    if context.distributed:
        barrier_devices = (
            [context.local_rank] if context.device.type == "cuda" else None
        )
        dist.barrier(device_ids=barrier_devices)

    if context.uses_fsdp:
        model_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optimizer_policy = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            model_policy,
            optimizer_policy,
        ):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = unwrap_model(model).state_dict()
        optimizer_state = optimizer.state_dict()

    if context.is_main:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "best_val_mse": best_val_mse,
                "epochs_without_improvement": epochs_without_improvement,
            },
            path,
        )
    if context.distributed:
        dist.barrier(device_ids=barrier_devices)


def run_smoke_test() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(2, 1, 1, 8, 16, device=device)
    targets = torch.randn(2, 1, 8, 16, device=device)
    cases = (
        ("baseline", False, False, 32, 32),
        ("compression_only", True, False, 8, 8),
        ("token_dropping_only", False, True, 32, 8),
        ("compression_and_token_dropping", True, True, 8, 2),
    )

    for name, compression_enabled, token_dropping, num_patches, num_active in cases:
        model = SparseReslim(
            variables=("2m_temperature",),
            output_variables=("2m_temperature",),
            img_size=(8, 16),
            patch_size=2,
            embed_dim=32,
            depth=4,
            num_heads=4,
            keep_ratio=0.25,
            num_dense_early=1,
            num_sparse_middle=2,
            token_dropping=token_dropping,
            compression_enabled=compression_enabled,
            compress_ratio=2,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_mse = _run_epoch(
            model,
            [(inputs, targets)],
            device,
            "float32",
            optimizer=optimizer,
        )
        forecast = model(inputs)
        assert forecast.shape == targets.shape
        assert model.num_patches == num_patches
        assert model.last_sparse_token_count == num_active
        print(
            f"Smoke test passed ({name}):",
            f"device={device},",
            f"forecast={tuple(forecast.shape)},",
            f"active_tokens={model.last_sparse_token_count}/{model.num_patches},",
            f"train_mse={train_mse:.4f}",
        )

    full_inputs = torch.randn(1, 1, 1, 12, 24, device=device)
    full_targets = torch.randn(1, 1, 12, 24, device=device)
    specs = build_tile_specs((12, 24), div=2, overlap=2)
    identity_tiles = [extract_tile(full_inputs, spec) for spec in specs]
    assert torch.equal(stitch_tiles(identity_tiles, specs, (12, 24)), full_inputs)

    tiled_model = SparseReslim(
        variables=("2m_temperature",),
        output_variables=("2m_temperature",),
        img_size=specs[0].shape,
        patch_size=2,
        embed_dim=32,
        depth=4,
        num_heads=4,
        keep_ratio=0.25,
        num_dense_early=1,
        num_sparse_middle=2,
        token_dropping=True,
        compression_enabled=True,
        compress_ratio=2,
    ).to(device)
    optimizer = torch.optim.AdamW(tiled_model.parameters(), lr=1e-3)
    tile_batches = [
        (extract_tile(full_inputs, spec), extract_tile(full_targets, spec))
        for spec in specs
    ]
    train_mse = _run_epoch(
        tiled_model,
        tile_batches,
        device,
        "float32",
        optimizer=optimizer,
    )
    tiled_model.eval()
    with torch.no_grad():
        predictions = [
            tiled_model(extract_tile(full_inputs, spec)) for spec in specs
        ]
    stitched = stitch_tiles(predictions, specs, (12, 24))
    assert stitched.shape == full_targets.shape
    assert tiled_model.last_sparse_token_count == 2
    print(
        "TILES smoke test passed:",
        f"tiles={len(specs)},",
        f"tile_shape={specs[0].shape},",
        f"stitched={tuple(stitched.shape)},",
        "compression=on, token_dropping=on,",
        f"train_mse={train_mse:.4f}",
    )


def _run_training(args, context: DistributedContext) -> None:
    device = context.device
    seed_everything(args.seed + context.rank)
    root = args.era5_dir.expanduser().resolve()
    full_img_size = _infer_image_size(root, args.input_vars[0])
    try:
        img_size = _resolve_model_image_size(full_img_size, args.tiling)
    except ValueError as error:
        raise SystemExit(f"Invalid TILES configuration: {error}") from error
    compress_ratio = (
        args.compression["compress_ratio"] if args.compression["enabled"] else 1
    )
    if img_size[0] % compress_ratio or img_size[1] % compress_ratio:
        raise SystemExit(
            f"Image size {img_size} is not divisible by compression ratio "
            f"{compress_ratio}"
        )
    compressed_img_size = (
        img_size[0] // compress_ratio,
        img_size[1] // compress_ratio,
    )
    if (
        compressed_img_size[0] % args.patch_size
        or compressed_img_size[1] % args.patch_size
    ):
        raise SystemExit(
            f"Compressed image size {compressed_img_size} is not divisible by "
            f"patch size {args.patch_size}"
        )

    common_dataset_args = dict(
        root=root,
        input_variables=args.input_vars,
        output_variables=args.output_vars,
        history=args.history,
        window=args.window,
        pred_range=args.pred_range,
        do_tiling=args.tiling["do_tiling"],
        tile_div=args.tiling["div"],
        tile_overlap=args.tiling["overlap"],
        rank=context.rank,
        world_size=context.world_size,
    )
    train_dataset = ERA5ForecastDataset(
        split="train", shuffle_files=True, **common_dataset_args
    )
    val_dataset = ERA5ForecastDataset(
        split="val", shuffle_files=False, **common_dataset_args
    )
    test_dataset = ERA5ForecastDataset(
        split="test", shuffle_files=False, **common_dataset_args
    )

    def loader(dataset):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    raw_network = _build_model(args, img_size)
    active_tokens = raw_network.num_patches
    if args.token_dropping and args.num_sparse_middle:
        active_tokens = max(1, int(raw_network.num_patches * args.keep_ratio))
    num_patches = raw_network.num_patches
    network = wrap_model(raw_network, args, context)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = args.output_dir.expanduser().resolve()
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    start_epoch = 0
    best_val_mse = float("inf")
    epochs_without_improvement = 0

    if args.pretrain:
        _load_checkpoint(
            Path(args.pretrain).expanduser(), network, context=context
        )
        if context.is_main:
            print(f"Loaded pretrained weights from {args.pretrain}", flush=True)
    if args.checkpoint:
        checkpoint = _load_checkpoint(
            Path(args.checkpoint).expanduser(),
            network,
            optimizer,
            context,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_mse = float(checkpoint.get("best_val_mse", best_val_mse))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        if context.is_main:
            print(f"Resuming at epoch {start_epoch + 1}", flush=True)

    if context.is_main:
        print(
            f"Training on {device} with native PyTorch | "
            f"parallelism={context.mode} ({context.world_size} processes) | "
            f"activation_checkpointing="
            f"{'on' if args.parallelism['activation_checkpointing'] else 'off'} | "
            f"tiling={'on' if args.tiling['do_tiling'] else 'off'} "
            f"(div={args.tiling['div']}, overlap={args.tiling['overlap']}, "
            f"tile={img_size}) | "
            f"compression={'on' if args.compression['enabled'] else 'off'} "
            f"(ratio={compress_ratio}) | "
            f"token_dropping={'on' if args.token_dropping else 'off'} "
            f"(keep_ratio={args.keep_ratio}, "
            f"active_tokens={active_tokens}/{num_patches})",
            flush=True,
        )
    for epoch in range(start_epoch, args.max_epochs):
        train_mse = _run_epoch(
            network,
            loader(train_dataset),
            device,
            args.data_type,
            optimizer=optimizer,
            max_batches=args.limit_train_batches,
        )
        val_mse = _run_epoch(
            network,
            loader(val_dataset),
            device,
            args.data_type,
            max_batches=args.limit_val_batches,
        )
        if context.is_main:
            print(
                f"Epoch {epoch + 1:03d}/{args.max_epochs:03d} "
                f"train/mse={train_mse:.6f} val/mse={val_mse:.6f}",
                flush=True,
            )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint,
                network,
                optimizer,
                epoch,
                best_val_mse,
                epochs_without_improvement,
                context,
            )
            if context.is_main:
                print(f"Saved best checkpoint to {best_checkpoint}", flush=True)
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                if context.is_main:
                    print(
                        f"Early stopping after {epochs_without_improvement} "
                        "epochs without validation improvement",
                        flush=True,
                    )
                break

    if not best_checkpoint.exists():
        raise RuntimeError(
            "Training completed without producing a best checkpoint. "
            "Check max_epochs and checkpoint settings."
        )
    _load_checkpoint(best_checkpoint, network, context=context)
    test_mse = _run_epoch(
        network,
        loader(test_dataset),
        device,
        args.data_type,
        max_batches=args.limit_test_batches,
    )
    if context.is_main:
        print(f"test/mse={test_mse:.6f}", flush=True)


def run_training(args) -> None:
    context = None
    try:
        context = initialize_distributed(args)
        _run_training(args, context)
    finally:
        cleanup_distributed(context)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Minimal Sparse-Reslim deterministic ERA5 forecasting example"
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="STORM-style YAML configuration file",
    )
    parser.add_argument("--smoke-test", action="store_true")
    cli_args = parser.parse_args(argv)
    if cli_args.smoke_test:
        return cli_args
    if cli_args.config is None:
        parser.error("CONFIG is required unless --smoke-test is used")
    try:
        return load_forecast_config(cli_args.config)
    except ConfigError as error:
        parser.error(str(error))


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        run_smoke_test()
    else:
        run_training(arguments)
