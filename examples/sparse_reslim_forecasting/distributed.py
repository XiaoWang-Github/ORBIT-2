"""Native PyTorch distributed setup for the forecasting example."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DistributedContext:
    """Process identity and process groups derived from the YAML and launcher."""

    rank: int
    local_rank: int
    world_size: int
    mode: str
    device: torch.device
    fsdp_group: Any = None
    ddp_group: Any = None

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def uses_fsdp(self) -> bool:
        return self.mode in {"fsdp", "hybrid_fsdp"}


def expected_world_size(parallelism: Mapping[str, Any]) -> int:
    """Return the number of processes required by the supported dimensions."""

    return (
        int(parallelism["fsdp"])
        * int(parallelism["simple_ddp"])
        * int(parallelism["tensor_par"])
        * int(parallelism["seq_par"])
    )


def parallelism_mode(parallelism: Mapping[str, Any]) -> str:
    """Map the YAML dimensions to the native PyTorch wrapping strategy."""

    if parallelism["tensor_par"] != 1 or parallelism["seq_par"] != 1:
        raise ValueError(
            "The forecasting example currently supports tensor_par=1 and "
            "seq_par=1"
        )
    fsdp_size = int(parallelism["fsdp"])
    ddp_size = int(parallelism["simple_ddp"])
    if fsdp_size > 1 and ddp_size > 1:
        return "hybrid_fsdp"
    if fsdp_size > 1:
        return "fsdp"
    if ddp_size > 1:
        return "ddp"
    return "single"


def _distributed_ranks() -> tuple[int, int, int]:
    if "SLURM_NTASKS" in os.environ:
        return (
            int(os.environ["SLURM_NTASKS"]),
            int(os.environ["SLURM_PROCID"]),
            int(os.environ["SLURM_LOCALID"]),
        )
    return (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def _create_hybrid_groups(
    fsdp_size: int, ddp_size: int, rank: int
) -> tuple[Any, Any]:
    """Create orthogonal shard and replica groups in a stable global order."""

    fsdp_group = None
    ddp_group = None
    for replica_index in range(ddp_size):
        ranks = [replica_index * fsdp_size + index for index in range(fsdp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            fsdp_group = group
    for shard_index in range(fsdp_size):
        ranks = [
            replica_index * fsdp_size + shard_index
            for replica_index in range(ddp_size)
        ]
        group = dist.new_group(ranks)
        if rank in ranks:
            ddp_group = group
    return fsdp_group, ddp_group


def initialize_distributed(config) -> DistributedContext:
    """Initialize one process per GPU from Slurm or torchrun environment."""

    mode = parallelism_mode(config.parallelism)
    expected_size = expected_world_size(config.parallelism)
    world_size, rank, local_rank = _distributed_ranks()
    if config.devices != expected_size:
        raise RuntimeError(
            f"trainer.devices={config.devices}, but the parallelism dimensions "
            f"require {expected_size} processes"
        )
    if world_size != expected_size:
        raise RuntimeError(
            f"The launcher provided {world_size} processes, but the YAML requires "
            f"{expected_size}"
        )

    if config.accelerator == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif config.accelerator == "gpu":
        raise RuntimeError(
            "trainer.accelerator is 'gpu', but PyTorch cannot access a GPU"
        )
    else:
        device = torch.device("cpu")

    if world_size > 1:
        if not os.environ.get("MASTER_ADDR") or not os.environ.get("MASTER_PORT"):
            raise RuntimeError(
                "MASTER_ADDR and MASTER_PORT must be set by the distributed launcher"
            )
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(hours=2),
        )

    fsdp_group = None
    ddp_group = None
    if mode == "hybrid_fsdp":
        fsdp_group, ddp_group = _create_hybrid_groups(
            int(config.parallelism["fsdp"]),
            int(config.parallelism["simple_ddp"]),
            rank,
        )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        mode=mode,
        device=device,
        fsdp_group=fsdp_group,
        ddp_group=ddp_group,
    )


def _apply_activation_checkpointing(model: nn.Module) -> None:
    wrapper = lambda module: checkpoint_wrapper(
        module,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, nn.TransformerEncoderLayer),
    )


def wrap_model(model: nn.Module, config, context: DistributedContext) -> nn.Module:
    """Apply activation checkpointing and the configured data parallel wrapper."""

    checkpoint_activations = bool(
        config.parallelism["activation_checkpointing"]
    )
    if context.mode in {"single", "ddp"} and checkpoint_activations:
        _apply_activation_checkpointing(model)

    if context.mode == "single":
        return model.to(context.device)
    if context.mode == "ddp":
        model = model.to(context.device)
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        return DDP(model, device_ids=device_ids, broadcast_buffers=False)

    mixed_precision = None
    if config.data_type == "bfloat16":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    auto_wrap_policy = lambda module, recurse, nonwrapped_numel: (
        transformer_auto_wrap_policy(
            module,
            recurse,
            nonwrapped_numel,
            transformer_layer_cls={nn.TransformerEncoderLayer},
        )
    )
    if context.mode == "hybrid_fsdp":
        process_group = (context.fsdp_group, context.ddp_group)
        strategy = ShardingStrategy.HYBRID_SHARD
    else:
        process_group = None
        strategy = ShardingStrategy.FULL_SHARD
    wrapped = FSDP(
        model,
        device_id=context.local_rank,
        process_group=process_group,
        sync_module_states=True,
        sharding_strategy=strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    if checkpoint_activations:
        _apply_activation_checkpointing(wrapped)
    return wrapped


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the user model from DDP; FSDP state handling stays collective."""

    return model.module if isinstance(model, DDP) else model


def cleanup_distributed(context: DistributedContext | None) -> None:
    if context is not None and context.distributed and dist.is_initialized():
        dist.destroy_process_group()
