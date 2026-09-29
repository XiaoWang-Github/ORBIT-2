"""Train the minimal Sparse-Reslim deterministic forecasting example."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from .model import SparseReslim
except ImportError:  # Support `python examples/.../train.py` from the repo root.
    from model import SparseReslim


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
        means = {name: float(np.asarray(mean_file[name]).reshape(-1)[0]) for name in variables}
        stds = {name: float(np.asarray(std_file[name]).reshape(-1)[0]) for name in variables}
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

    def __iter__(self):
        files = list(self.files)
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
                yield torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(
                    targets.astype(np.float32)
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
        keep_ratio=args.keep_ratio,
        num_dense_early=args.num_dense_early,
        num_sparse_middle=args.num_sparse_middle,
    )


def run_smoke_test() -> None:
    torch.manual_seed(0)
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
    )
    inputs = torch.randn(2, 1, 1, 8, 16, requires_grad=True)
    targets = torch.randn(2, 1, 8, 16)
    forecast = model(inputs)
    loss = F.mse_loss(forecast, targets)
    loss.backward()
    assert forecast.shape == targets.shape
    assert model.last_sparse_token_count == 8  # 25% of 32 patch tokens.
    print(
        "Smoke test passed:",
        f"forecast={tuple(forecast.shape)},",
        f"sparse_tokens={model.last_sparse_token_count}/{model.num_patches},",
        f"loss={loss.item():.4f}",
    )


def run_training(args) -> None:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError as error:
        raise SystemExit(
            "PyTorch Lightning is required for training. Run `pip install -e .` "
            "from the ORBIT-2 repository root."
        ) from error

    if args.era5_dir is None:
        raise SystemExit("ERA5_DIR is required unless --smoke-test is used")
    root = args.era5_dir.expanduser().resolve()
    img_size = _infer_image_size(root, args.input_vars[0])
    if img_size[0] % args.patch_size or img_size[1] % args.patch_size:
        raise SystemExit(
            f"Image size {img_size} is not divisible by patch size {args.patch_size}"
        )

    common_dataset_args = dict(
        root=root,
        input_variables=args.input_vars,
        output_variables=args.output_vars,
        history=args.history,
        window=args.window,
        pred_range=args.pred_range,
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
            pin_memory=args.accelerator != "cpu",
        )

    network = _build_model(args, img_size)

    class ForecastModule(pl.LightningModule):
        def __init__(self, net: SparseReslim):
            super().__init__()
            self.net = net

        def forward(self, inputs):
            return self.net(inputs)

        def _shared_step(self, batch, stage: str):
            inputs, targets = batch
            loss = F.mse_loss(self(inputs), targets)
            self.log(
                f"{stage}/mse",
                loss,
                prog_bar=True,
                on_step=stage == "train",
                on_epoch=True,
            )
            return loss

        def training_step(self, batch, batch_idx):
            return self._shared_step(batch, "train")

        def validation_step(self, batch, batch_idx):
            self._shared_step(batch, "val")

        def test_step(self, batch, batch_idx):
            self._shared_step(batch, "test")

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

    pl.seed_everything(args.seed, workers=True)
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        monitor="val/mse",
        mode="min",
        filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=1,
    )
    callbacks = [checkpoint]
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor="val/mse", patience=args.patience))

    trainer_kwargs = {}
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        default_root_dir=output_dir,
        callbacks=callbacks,
        **trainer_kwargs,
    )
    module = ForecastModule(network)
    trainer.fit(module, train_dataloaders=loader(train_dataset), val_dataloaders=loader(val_dataset))
    trainer.test(module, dataloaders=loader(test_dataset), ckpt_path="best")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal Sparse-Reslim deterministic ERA5 forecasting example"
    )
    parser.add_argument("era5_dir", nargs="?", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--input-vars", nargs="+", default=["2m_temperature"])
    parser.add_argument("--output-vars", nargs="+", default=["2m_temperature"])
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--pred-range", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--num-dense-early", type=int, default=1)
    parser.add_argument("--num-sparse-middle", type=int, default=4)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sparse_reslim_forecasting"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        run_smoke_test()
    else:
        run_training(arguments)
