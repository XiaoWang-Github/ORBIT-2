"""Shared utility functions for ORBIT-2 training and visualization scripts.

This module contains common utilities used across different scripts to avoid
code duplication and ensure consistent behavior.

Functions:
    seed_everything: Set random seeds for reproducibility
    init_par_groups: Initialize distributed parallel process groups
"""

import os
import random
import torch
import numpy as np
import torch.distributed as dist


def seed_everything(seed):
    """Set random seeds for reproducibility across all libraries.

    This function sets random seeds for Python's random module, NumPy,
    and PyTorch (both CPU and CUDA) to ensure reproducible results.
    It also sets PyTorch's cuDNN to deterministic mode.

    Args:
        seed (int): Random seed value to use

    Note:
        Setting cuDNN to deterministic mode may impact performance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_par_groups(
    data_par_size, tensor_par_size, seq_par_size, fsdp_size, simple_ddp_size, num_heads
):
    """Initialize distributed parallel process groups for model training.

    This function creates the necessary process groups for different types of
    parallelism used in distributed training of large models.

    Args:
        data_par_size (int): Size of data parallelism groups
        tensor_par_size (int): Size of tensor parallelism groups
        seq_par_size (int): Size of sequence parallelism groups (must be 1, not implemented)
        fsdp_size (int): Size of FSDP (Fully Sharded Data Parallel) groups
        simple_ddp_size (int): Size of simple DDP groups
        num_heads (int): Number of attention heads in the model

    Returns:
        tuple: (seq_par_group, data_par_group, tensor_par_group,
                data_seq_ort_group, fsdp_group, simple_ddp_group)

    Raises:
        AssertionError: If parallelism configuration is invalid
    """
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()

    # Validate parallelism configuration
    assert seq_par_size == 1, "Sequence parallelism not implemented"

    assert (
        data_par_size * seq_par_size * tensor_par_size
    ) == world_size, (
        "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
    )
    assert (
        num_heads % tensor_par_size
    ) == 0, "model heads % tensor parallel size must be 0"

    # Initialize tensor parallelism group
    tensor_par_group = None

    for i in range(data_par_size * seq_par_size):
        ranks = [j for j in range(i * tensor_par_size, (i + 1) * tensor_par_size)]

        if world_rank == 0:
            print(
                "i ",
                i,
                " data_par_size ",
                data_par_size,
                " SEQ_PAR_SIZE ",
                seq_par_size,
                " TENSOR_PAR_SIZE ",
                tensor_par_size,
                " tensor_par_group ranks ",
                ranks,
            )

        group = dist.new_group(ranks)

        if world_rank in ranks:
            tensor_par_group = group

    # Initialize sequence parallelism group (currently not used)
    seq_par_group = None

    for t in range(data_par_size):
        for i in range(tensor_par_size):
            ranks = [
                t * tensor_par_size * seq_par_size + i + j * tensor_par_size
                for j in range(seq_par_size)
            ]

            if world_rank == 0:
                print(
                    "i ",
                    i,
                    " data_par_size ",
                    data_par_size,
                    " SEQ_PAR_SIZE ",
                    seq_par_size,
                    " TENSOR_PAR_SIZE ",
                    tensor_par_size,
                    " seq_par_group ranks ",
                    ranks,
                    flush=True,
                )

            group = dist.new_group(ranks)

            if world_rank in ranks:

                seq_par_group = group

    # Initialize data parallelism and FSDP groups
    data_par_group = None
    fsdp_group = None
    simple_ddp_group = None

    for i in range(tensor_par_size * seq_par_size):
        ranks = [i + j * tensor_par_size * seq_par_size for j in range(data_par_size)]

        for k in range(simple_ddp_size):
            fsdp_begin_idx = k * fsdp_size
            fsdp_end_idx = (k + 1) * fsdp_size
            fsdp_ranks = ranks[fsdp_begin_idx:fsdp_end_idx]

            if world_rank == 0:
                print(
                    "i ",
                    i,
                    " data_par_size ",
                    data_par_size,
                    " SEQ_PAR_SIZE ",
                    seq_par_size,
                    " TENSOR_PAR_SIZE ",
                    tensor_par_size,
                    " fsdp_ranks",
                    fsdp_ranks,
                )

            group = dist.new_group(fsdp_ranks)
            if world_rank in fsdp_ranks:
                fsdp_group = group

        for k in range(fsdp_size):
            simple_ddp_begin_idx = k
            simple_ddp_end_idx = len(ranks)
            simple_ddp_ranks = ranks[simple_ddp_begin_idx:simple_ddp_end_idx:fsdp_size]

            if world_rank == 0:
                print(
                    "i ",
                    i,
                    " data_par_size ",
                    data_par_size,
                    " SEQ_PAR_SIZE ",
                    seq_par_size,
                    " TENSOR_PAR_SIZE ",
                    tensor_par_size,
                    " simple_ddp_ranks",
                    simple_ddp_ranks,
                )

            group = dist.new_group(simple_ddp_ranks)
            if world_rank in simple_ddp_ranks:
                simple_ddp_group = group

        if world_rank == 0:
            print(
                "i ",
                i,
                " data_par_size ",
                data_par_size,
                " SEQ_PAR_SIZE ",
                seq_par_size,
                " TENSOR_PAR_SIZE ",
                tensor_par_size,
                " data_par_group ranks ",
                ranks,
            )
        group = dist.new_group(ranks)
        if world_rank in ranks:
            data_par_group = group

    # Initialize orthogonal group for data and sequence parallelism
    data_seq_ort_group = None

    for i in range(tensor_par_size):
        ranks = [i + tensor_par_size * j for j in range(data_par_size * seq_par_size)]

        if world_rank == 0:
            print(
                "i ",
                i,
                " data_par_size ",
                data_par_size,
                " SEQ_PAR_SIZE ",
                seq_par_size,
                " TENSOR_PAR_SIZE ",
                tensor_par_size,
                " data_seq_ort_group ranks ",
                ranks,
            )
        group = dist.new_group(ranks)

        if world_rank in ranks:
            data_seq_ort_group = group

    return (
        seq_par_group,
        data_par_group,
        tensor_par_group,
        data_seq_ort_group,
        fsdp_group,
        simple_ddp_group,
    )
