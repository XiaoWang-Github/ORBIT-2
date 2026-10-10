"""Configuration loading for the Sparse-Reslim forecasting example."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a forecasting configuration is missing or inconsistent."""


@dataclass(frozen=True)
class ForecastConfig:
    """Validated, flattened view of the STORM-style forecasting YAML."""

    config_path: Path
    dataset_name: str
    source: str
    era5_dir: Path
    input_vars: tuple[str, ...]
    output_vars: tuple[str, ...]
    history: int
    window: int
    pred_range: int

    max_epochs: int
    batch_size: int
    num_workers: int
    patience: int
    accelerator: str
    devices: int
    output_dir: Path
    limit_train_batches: int | None
    limit_val_batches: int | None
    limit_test_batches: int | None
    seed: int
    checkpoint: str | None
    pretrain: str | None
    data_type: str
    gpu_type: str
    train_loss: str

    preset: str
    lr: float
    weight_decay: float
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    keep_ratio: float
    num_dense_early: int
    num_sparse_middle: int
    token_dropping: bool

    parallelism: Mapping[str, Any]
    tiling: Mapping[str, Any]
    compression: Mapping[str, Any]
    smoke_test: bool = False


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name!r} must be a mapping")
    return value


def _required(section: Mapping[str, Any], key: str, section_name: str) -> Any:
    if key not in section or section[key] is None:
        raise ConfigError(f"Missing required value: {section_name}.{key}")
    return section[key]


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be {qualifier}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _dataset_value(
    section: Mapping[str, Any], key: str, dataset_name: str
) -> Any:
    values = _section(section, key)
    if dataset_name not in values:
        raise ConfigError(f"data.{key} is missing dataset {dataset_name!r}")
    return values[dataset_name]


def _variable_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ConfigError(f"{name} must contain non-empty variable names")
    return tuple(value)


def _expand_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(f"{name} must be a filesystem path")
    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def load_forecast_config(path: str | os.PathLike[str]) -> ForecastConfig:
    """Load and validate a single-dataset forecasting YAML configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigError(f"Unable to read config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw, Mapping):
        raise ConfigError("The YAML document must contain a top-level mapping")

    trainer = _section(raw, "trainer")
    parallelism = _section(raw, "parallelism")
    tiling = _section(raw, "tiling")
    compression = _section(raw, "compression")
    model = _section(raw, "model")
    data = _section(raw, "data")

    data_dirs = _section(data, "data_dir")
    if len(data_dirs) != 1:
        raise ConfigError(
            "The current forecasting example supports exactly one data.data_dir entry"
        )
    dataset_name = next(iter(data_dirs))

    input_vars = _variable_list(
        _dataset_value(data, "dict_in_variables", dataset_name),
        f"data.dict_in_variables.{dataset_name}",
    )
    output_vars = _variable_list(
        _dataset_value(data, "dict_out_variables", dataset_name),
        f"data.dict_out_variables.{dataset_name}",
    )
    unknown_outputs = set(output_vars) - set(input_vars)
    if unknown_outputs:
        raise ConfigError(
            "Output variables must also be inputs for the residual path: "
            f"{sorted(unknown_outputs)}"
        )

    history = _positive_int(
        _dataset_value(data, "history", dataset_name), "data.history"
    )
    window = _positive_int(
        _dataset_value(data, "window", dataset_name), "data.window"
    )
    pred_range = _positive_int(
        _dataset_value(data, "pred_range", dataset_name), "data.pred_range"
    )

    depth = _positive_int(_required(model, "depth", "model"), "model.depth")
    num_dense_early = _positive_int(
        _required(model, "num_dense_early", "model"),
        "model.num_dense_early",
        allow_zero=True,
    )
    num_sparse_middle = _positive_int(
        _required(model, "num_sparse_middle", "model"),
        "model.num_sparse_middle",
        allow_zero=True,
    )
    if num_dense_early + num_sparse_middle > depth:
        raise ConfigError(
            "model.num_dense_early + model.num_sparse_middle must not exceed "
            "model.depth"
        )

    embed_dim = _positive_int(
        _required(model, "embed_dim", "model"), "model.embed_dim"
    )
    num_heads = _positive_int(
        _required(model, "num_heads", "model"), "model.num_heads"
    )
    if embed_dim % num_heads:
        raise ConfigError("model.embed_dim must be divisible by model.num_heads")

    keep_ratio = float(_required(model, "keep_ratio", "model"))
    if not 0 < keep_ratio <= 1:
        raise ConfigError("model.keep_ratio must be in (0, 1]")
    token_dropping = _boolean(
        _required(model, "token_dropping", "model"),
        "model.token_dropping",
    )

    validated_parallelism = {
        key: _positive_int(
            _required(parallelism, key, "parallelism"), f"parallelism.{key}"
        )
        for key in ("fsdp", "simple_ddp", "tensor_par", "seq_par")
    }
    validated_parallelism["activation_checkpointing"] = _boolean(
        parallelism.get("activation_checkpointing", False),
        "parallelism.activation_checkpointing",
    )
    if (
        validated_parallelism["tensor_par"] != 1
        or validated_parallelism["seq_par"] != 1
    ):
        raise ConfigError(
            "The forecasting example currently requires parallelism.tensor_par=1 "
            "and parallelism.seq_par=1"
        )
    do_tiling = _boolean(
        _required(tiling, "do_tiling", "tiling"), "tiling.do_tiling"
    )
    validated_tiling = {
        "do_tiling": do_tiling,
        "div": _positive_int(_required(tiling, "div", "tiling"), "tiling.div"),
        "overlap": _positive_int(
            _required(tiling, "overlap", "tiling"),
            "tiling.overlap",
            allow_zero=True,
        ),
    }
    if do_tiling and validated_tiling["div"] == 1:
        raise ConfigError("tiling.div must be greater than 1 when tiling is enabled")
    compression_enabled = _boolean(
        _required(compression, "enabled", "compression"),
        "compression.enabled",
    )
    compress_ratio = _positive_int(
        _required(compression, "compress_ratio", "compression"),
        "compression.compress_ratio",
    )
    if compression_enabled and compress_ratio == 1:
        raise ConfigError(
            "compression.compress_ratio must be greater than 1 when "
            "compression is enabled"
        )
    validated_compression = {
        "enabled": compression_enabled,
        "compress_ratio": compress_ratio,
    }

    accelerator = str(trainer.get("accelerator", "auto"))
    if accelerator not in {"auto", "cpu", "gpu"}:
        raise ConfigError("trainer.accelerator must be one of: auto, cpu, gpu")
    data_type = str(_required(trainer, "data_type", "trainer"))
    if data_type not in {"bfloat16", "float32"}:
        raise ConfigError("trainer.data_type must be one of: bfloat16, float32")
    train_loss = str(_required(trainer, "train_loss", "trainer"))
    if train_loss != "mse":
        raise ConfigError("The forecasting example currently supports train_loss: mse")
    if trainer.get("checkpoint") and trainer.get("pretrain"):
        raise ConfigError("Set only one of trainer.checkpoint or trainer.pretrain")

    devices = _positive_int(trainer.get("devices", 1), "trainer.devices")
    expected_devices = (
        validated_parallelism["fsdp"]
        * validated_parallelism["simple_ddp"]
        * validated_parallelism["tensor_par"]
        * validated_parallelism["seq_par"]
    )
    if devices != expected_devices:
        raise ConfigError(
            f"trainer.devices={devices}, but parallelism requires "
            f"{expected_devices} processes"
        )

    batch_limits = {}
    for stage in ("train", "val", "test"):
        key = f"limit_{stage}_batches"
        value = trainer.get(key)
        if value is not None:
            value = _positive_int(value, f"trainer.{key}")
        batch_limits[key] = value

    return ForecastConfig(
        config_path=config_path,
        dataset_name=dataset_name,
        source=str(_dataset_value(data, "src", dataset_name)),
        era5_dir=_expand_path(data_dirs[dataset_name], "data.data_dir"),
        input_vars=input_vars,
        output_vars=output_vars,
        history=history,
        window=window,
        pred_range=pred_range,
        max_epochs=_positive_int(
            _required(trainer, "max_epochs", "trainer"), "trainer.max_epochs"
        ),
        batch_size=_positive_int(
            _required(trainer, "batch_size", "trainer"), "trainer.batch_size"
        ),
        num_workers=_positive_int(
            _required(trainer, "num_workers", "trainer"),
            "trainer.num_workers",
            allow_zero=True,
        ),
        patience=_positive_int(
            trainer.get("patience", 0), "trainer.patience", allow_zero=True
        ),
        accelerator=accelerator,
        devices=devices,
        output_dir=_expand_path(
            trainer.get("output_dir", "outputs/sparse_reslim_forecasting"),
            "trainer.output_dir",
        ),
        limit_train_batches=batch_limits["limit_train_batches"],
        limit_val_batches=batch_limits["limit_val_batches"],
        limit_test_batches=batch_limits["limit_test_batches"],
        seed=_positive_int(trainer.get("seed", 0), "trainer.seed", allow_zero=True),
        checkpoint=trainer.get("checkpoint"),
        pretrain=trainer.get("pretrain"),
        data_type=data_type,
        gpu_type=str(_required(trainer, "gpu_type", "trainer")),
        train_loss=train_loss,
        preset=str(_required(model, "preset", "model")),
        lr=float(_required(model, "lr", "model")),
        weight_decay=float(_required(model, "weight_decay", "model")),
        patch_size=_positive_int(
            _required(model, "patch_size", "model"), "model.patch_size"
        ),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=float(model.get("mlp_ratio", 4.0)),
        dropout=float(model.get("dropout", 0.0)),
        keep_ratio=keep_ratio,
        num_dense_early=num_dense_early,
        num_sparse_middle=num_sparse_middle,
        token_dropping=token_dropping,
        parallelism=validated_parallelism,
        tiling=validated_tiling,
        compression=validated_compression,
    )
