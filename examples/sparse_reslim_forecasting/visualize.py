"""Visualize Sparse-Reslim forecasting inputs, targets, and predictions."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .config import ConfigError, load_forecast_config
    from .train import (
        ERA5ForecastDataset,
        _autocast_context,
        _build_model,
        _infer_image_size,
        _load_checkpoint,
        _resolve_model_image_size,
        _select_device,
    )
    from .tiling import build_tile_specs, extract_tile, stitch_tiles
    from .utils import seed_everything
except ImportError:  # Support `python examples/.../visualize.py`.
    from config import ConfigError, load_forecast_config
    from train import (
        ERA5ForecastDataset,
        _autocast_context,
        _build_model,
        _infer_image_size,
        _load_checkpoint,
        _resolve_model_image_size,
        _select_device,
    )
    from tiling import build_tile_specs, extract_tile, stitch_tiles
    from utils import seed_everything


def _non_negative_int(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return integer


def _load_sample(config, split: str, sample_index: int):
    dataset = ERA5ForecastDataset(
        root=config.era5_dir.expanduser().resolve(),
        split=split,
        input_variables=config.input_vars,
        output_variables=config.output_vars,
        history=config.history,
        window=config.window,
        pred_range=config.pred_range,
        shuffle_files=False,
    )
    sample = next(islice(iter(dataset), sample_index, sample_index + 1), None)
    if sample is None:
        raise IndexError(
            f"Sample index {sample_index} is outside the {split!r} dataset"
        )
    return dataset, sample


def _plot_forecast(
    input_field: np.ndarray,
    target_field: np.ndarray,
    prediction_field: np.ndarray,
    *,
    variable: str,
    pred_range: int,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    normalized_mse: float,
    native_rmse: float,
    output_path: Path,
) -> None:
    fields = np.stack((input_field, target_field, prediction_field))
    value_min = float(np.nanmin(fields))
    value_max = float(np.nanmax(fields))
    error = prediction_field - target_field
    error_limit = float(np.nanmax(np.abs(error)))
    if error_limit == 0:
        error_limit = 1.0

    unit = "K" if variable == "2m_temperature" else "native units"
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.4), constrained_layout=True)
    panels = (
        (input_field, "Input at t", "coolwarm", value_min, value_max),
        (
            target_field,
            f"Ground truth at t + {pred_range} h",
            "coolwarm",
            value_min,
            value_max,
        ),
        (
            prediction_field,
            f"Prediction at t + {pred_range} h",
            "coolwarm",
            value_min,
            value_max,
        ),
        (error, "Prediction - ground truth", "RdBu_r", -error_limit, error_limit),
    )

    for axis, (field, title, color_map, lower, upper) in zip(axes, panels):
        image = axis.imshow(
            field,
            cmap=color_map,
            vmin=lower,
            vmax=upper,
            origin="upper",
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("longitude index")
        axis.set_ylabel("latitude index")
        figure.colorbar(image, ax=axis, shrink=0.78, label=unit)

    figure.suptitle(
        f"Sparse-Reslim forecast: {variable}\n"
        f"input {input_shape} → output {output_shape} | "
        f"normalized MSE {normalized_mse:.6f} | "
        f"RMSE {native_rmse:.4f} {unit}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _forecast_full_image(model, inputs, config):
    if not config.tiling["do_tiling"]:
        return model(inputs)

    image_size = tuple(inputs.shape[-2:])
    specs = build_tile_specs(
        image_size,
        config.tiling["div"],
        config.tiling["overlap"],
    )
    predictions = [model(extract_tile(inputs, spec)) for spec in specs]
    return stitch_tiles(predictions, specs, image_size)


def create_visualization(
    config,
    *,
    checkpoint_path: Path,
    split: str,
    sample_index: int,
    variable: str,
    output_path: Path,
    accelerator: str | None = None,
) -> dict[str, float]:
    if variable not in config.output_vars:
        raise ValueError(
            f"Variable {variable!r} is not in output variables {config.output_vars}"
        )

    seed_everything(config.seed)
    device = _select_device(accelerator or config.accelerator)
    dataset, (inputs, targets) = _load_sample(config, split, sample_index)
    image_size = _infer_image_size(config.era5_dir, config.input_vars[0])
    model_image_size = _resolve_model_image_size(image_size, config.tiling)
    model = _build_model(config, model_image_size).to(device)
    _load_checkpoint(checkpoint_path.expanduser().resolve(), model)
    model.eval()

    input_batch = inputs.unsqueeze(0).to(device)
    target_batch = targets.unsqueeze(0).to(device)
    with torch.inference_mode(), _autocast_context(device, config.data_type):
        forecast = _forecast_full_image(model, input_batch, config)

    output_index = config.output_vars.index(variable)
    input_index = config.input_vars.index(variable)
    normalized_mse = float(
        F.mse_loss(
            forecast[:, output_index].float(),
            target_batch[:, output_index].float(),
        )
    )

    mean = dataset.means[variable]
    standard_deviation = dataset.stds[variable]
    input_field = (
        inputs[-1, input_index].numpy() * standard_deviation + mean
    )
    target_field = (
        targets[output_index].numpy() * standard_deviation + mean
    )
    prediction_field = (
        forecast[0, output_index].float().cpu().numpy() * standard_deviation
        + mean
    )
    native_rmse = float(
        np.sqrt(np.mean(np.square(prediction_field - target_field)))
    )

    _plot_forecast(
        input_field,
        target_field,
        prediction_field,
        variable=variable,
        pred_range=config.pred_range,
        input_shape=tuple(input_batch.shape),
        output_shape=tuple(forecast.shape),
        normalized_mse=normalized_mse,
        native_rmse=native_rmse,
        output_path=output_path,
    )
    print(f"Visualization saved to {output_path.resolve()}", flush=True)
    print(
        f"input={tuple(input_batch.shape)} target={tuple(target_batch.shape)} "
        f"prediction={tuple(forecast.shape)} normalized_mse={normalized_mse:.6f} "
        f"native_rmse={native_rmse:.4f} "
        f"tiles={config.tiling['div'] ** 2 if config.tiling['do_tiling'] else 1}",
        flush=True,
    )
    return {"normalized_mse": normalized_mse, "native_rmse": native_rmse}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize a Sparse-Reslim deterministic ERA5 forecast"
    )
    parser.add_argument("config", type=Path, help="Forecasting YAML configuration")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint path; defaults to CONFIG output_dir/checkpoints/best.pt",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-index", type=_non_negative_int, default=0)
    parser.add_argument("--variable", help="Output variable to plot")
    parser.add_argument("--output", type=Path, help="Output image path")
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"))
    arguments = parser.parse_args(argv)
    try:
        config = load_forecast_config(arguments.config)
    except ConfigError as error:
        parser.error(str(error))

    variable = arguments.variable or config.output_vars[0]
    checkpoint = arguments.checkpoint or (
        config.output_dir / "checkpoints" / "best.pt"
    )
    output = arguments.output or (
        config.output_dir
        / "visualizations"
        / f"{arguments.split}_sample_{arguments.sample_index}_{variable}.png"
    )
    return arguments, config, checkpoint, variable, output


if __name__ == "__main__":
    cli, forecast_config, checkpoint, selected_variable, output = parse_args()
    create_visualization(
        forecast_config,
        checkpoint_path=checkpoint,
        split=cli.split,
        sample_index=cli.sample_index,
        variable=selected_variable,
        output_path=output,
        accelerator=cli.accelerator,
    )
