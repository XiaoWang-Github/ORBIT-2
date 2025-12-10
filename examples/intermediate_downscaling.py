# Standard library
from argparse import ArgumentParser
import os
import torch
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import Sequential
from datetime import timedelta
import sys
import time
import yaml

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
)
from climate_learn.models.hub.components.vit_blocks import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock,
)
from climate_learn.utils.fused_attn import FusedAttn
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from climate_learn.dist.profile import *
from utils import seed_everything, init_par_groups

def debug_print(*args, **kwargs):
    # Check rank using dist or env vars (fallback)
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    
    # Print only for rank 0, or if all_ranks is True
    if rank == 0 or kwargs.pop("all_ranks", False):
        print(f"[DEBUG_RANK_{rank}]", *args, **kwargs, flush=True)

def validate_data_type(data_type):
    """Validate that data_type is either bfloat16 or float32.
    
    Args:
        data_type (str): Data type string from configuration
        
    Raises:
        ValueError: If data_type is not 'bfloat16' or 'float32'
    """
    valid_types = ["bfloat16", "float32"]
    if data_type not in valid_types:
        raise ValueError(
            f"Invalid data_type '{data_type}'. "
            f"Only {valid_types} are supported. "
            f"float16 is no longer supported due to numerical stability issues. "
            f"Please use 'bfloat16' for 16-bit training or 'float32' for full precision."
        )


def log_gpu_memory(device, message="", world_rank=None):
    """Log GPU memory usage with optional message and rank."""
    memory_gb = torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024
    if world_rank is not None:
        print(
            f"rank {world_rank} {message} torch.cuda.memory_reserved: {memory_gb:.2f}GB",
            flush=True,
        )
    else:
        print(f"{message} torch.cuda.memory_reserved: {memory_gb:.2f}GB", flush=True)


def get_tensor_parallel_checkpoint_path(base_path, rank, tensor_par_size):
    """Get checkpoint path for tensor parallel models."""
    if tensor_par_size > 1:
        return f"{base_path}_rank_{rank}"
    return base_path


def get_checkpoint_filename(save_path, epoch, world_rank, tensor_par_size):
    """Generate checkpoint filename for given epoch and rank."""
    base_filename = f"{save_path}/interm_epoch_{epoch}.ckpt"
    if tensor_par_size > 1:
        return f"{base_filename}_rank_{world_rank}"
    return base_filename


def load_checkpoint_pretrain(
    model,
    checkpoint_path,
    pretrain_path,
    cp_save_path,
    tensor_par_size=1,
    tensor_par_group=None,
):
    """
    Load model weights from checkpoint or pretrained model.

    This function handles three scenarios:
    1. Resume training from a checkpoint (loads model, optimizer, scheduler states)
    2. Initialize from a pretrained model (loads only model weights)
    3. Initialize model for tensor parallelism when no checkpoint exists

    Args:
        model: PyTorch model to load weights into
        checkpoint_path (str): Path to checkpoint file for resuming training
        pretrain_path (str): Path to pretrained model weights
        cp_save_path (str): Directory where checkpoints will be saved
        tensor_par_size (int): Size of tensor parallelism (default: 1)
        tensor_par_group: Process group for tensor parallelism (default: None)

    Returns:
        tuple: (model, start_epoch) where start_epoch is the epoch to resume from
    """
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    # load model checkpoint
    if checkpoint_path is not None and world_rank < tensor_par_size:

        checkpoint_path = get_tensor_parallel_checkpoint_path(
            checkpoint_path, world_rank, tensor_par_size
        )

        if os.path.exists(checkpoint_path):

            print(
                "world_rank",
                world_rank,
                "model resume from checkpoint",
                checkpoint_path,
                " Checkpoint path found.",
                flush=True,
            )

            map_location = "cpu"

            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint["model_state_dict"])

            del checkpoint
        else:
            print(
                "resume from checkpoint was set to True. "
                "But the checkpoint path does not exist.",
                flush=True,
            )
            sys.exit("checkpoint path does not exist")

    # load pretrained model
    if pretrain_path is not None and world_rank < tensor_par_size:
        if tensor_par_size > 1:
            pretrain_path = pretrain_path + "_" + "rank" + "_" + str(world_rank)

        if os.path.exists(pretrain_path):
            print(
                "world_rank",
                world_rank,
                "load pretrained model",
                pretrain_path,
                " Pretrain path found.",
                flush=True,
            )
            _load_pretrained_weights(model, pretrain_path, local_rank, world_rank)
        else:
            print(
                "resume from pretrained model was set to True. "
                "But the pretrained model path does not exist.",
                flush=True,
            )
            sys.exit("pretrain path does not exist")

    # initialize weights for tensor parallelism when training from scratch
    if pretrain_path is None and checkpoint_path is None and tensor_par_size > 1:
        if world_rank == 0:
            isExist = os.path.exists(cp_save_path)

            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(cp_save_path)
                print("The new checkpoint saving directory is created!")

            # Save initial model weights and distribute to all GPUs in the tensor
            # parallel group to synchronize model weights that do not belong to the
            # training block

            init_model_dict = {
                k: v
                for k, v in model.state_dict().items()
                if ("attn" not in k and "mlp" not in k and "var_agg" not in k)
            }

            print(
                "training from scratch and tensor_par_size>1. rank",
                world_rank,
                "init_model_dict.keys()",
                init_model_dict.keys(),
                flush=True,
            )

            torch.save(
                init_model_dict,
                cp_save_path + "/initial_" + str(dist.get_rank()) + ".pth",
            )

            del init_model_dict

        dist.barrier(device_ids=[local_rank])

        if world_rank != 0 and world_rank < tensor_par_size:

            # Load initial model weights and synchronize model weights that are not
            # in the training block among sequence parallel GPUs
            print("training from scratch. rank", world_rank, flush=True)

            map_location = "cpu"
            model.load_state_dict(
                torch.load(
                    cp_save_path + "/initial_" + str(0) + ".pth",
                    map_location=map_location,
                ),
                strict=False,
            )


def _load_pretrained_weights(model, pretrain_path, device, world_rank):
    map_location = "cpu"
    checkpoint = torch.load(pretrain_path, map_location=map_location)

    print("Loading pre-trained checkpoint from: %s" % pretrain_path)
    pretrain_model = checkpoint["model_state_dict"]

    del checkpoint

    state_dict = model.state_dict()

    if torch.distributed.get_rank() == 0:
        for k in list(pretrain_model.keys()):
            print(
                "Pretrained model before deletion. Name ",
                k,
                "shape",
                pretrain_model[k].shape,
                flush=True,
            )

    for k in list(
        pretrain_model.keys()
    ):  # in pre-train model weights, but not fine-tuning model
        if k not in state_dict.keys():
            print(f"Removing key {k} from pretrained checkpoint: no exist")
            del pretrain_model[k]
        elif (
            pretrain_model[k].shape != state_dict[k].shape
        ):  # if pre-train and fine-tune model weights dimension doesn't match
            if k == "pos_embed":
                print("interpolate positional embedding", flush=True)
                interpolate_pos_embed(model, pretrain_model, new_size=model.img_size)
            else:
                print(
                    f"Removing key {k} from pretrained checkpoint: no matching shape",
                    pretrain_model[k].shape,
                    state_dict[k].shape,
                )
                del pretrain_model[k]

    # load pre-trained model
    msg = model.load_state_dict(pretrain_model, strict=False)
    print(msg)
    del pretrain_model


"""
Setup sequence, data, tensor model, and sequence_plus_data parallel groups
"""


def clip_replace_constant(y, yhat, out_variables):

    prcp_index = out_variables.index("total_precipitation_24hr")
    for i in range(yhat.shape[1]):
        if i == prcp_index:
            torch.clamp_(yhat[:, prcp_index, :, :], min=0.0)

    for i in range(yhat.shape[1]):
        # if constant replace with ground-truth value
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat


def training_step(
    batch, batch_idx, net, device: int, var_weights, train_loss_metric
) -> torch.Tensor:
    print(f"[{dist.get_rank()}] Entered training_step", flush=True)

    # Check model parameters for NaNs/Infs at the start of training step
    for name, param in net.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            print(f"[{dist.get_rank()}] CRITICAL: NaN/Inf detected in gradient of parameter {name}", flush=True)
            # You might want to break or raise an error here if this is unexpected
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"[{dist.get_rank()}] CRITICAL: NaN/Inf detected in parameter {name}", flush=True)
            # You might want to break or raise an error here if this is unexpected

    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)

    print(f"[{dist.get_rank()}] training_step: Batch {batch_idx} - Input x shape: {x.shape}, dtype: {x.dtype}", flush=True)
    print(f"[{dist.get_rank()}] training_step: Input x stats: min={x.min()}, max={x.max()}, mean={x.mean()}, std={x.std()}", flush=True)
    print(f"[{dist.get_rank()}] training_step: Input x NaN/Inf: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}", flush=True)
    print(f"[{dist.get_rank()}] training_step: Batch {batch_idx} - Target y shape: {y.shape}, dtype: {y.dtype}", flush=True)
    print(f"[{dist.get_rank()}] training_step: Target y stats: min={y.min()}, max={y.max()}, mean={y.mean()}, std={y.std()}", flush=True)
    print(f"[{dist.get_rank()}] training_step: Target y NaN/Inf: NaN={torch.isnan(y).any()}, Inf={torch.isinf(y).any()}", flush=True)

    try:
        yhat = net.forward(x, in_variables, out_variables)
        print(f"[{dist.get_rank()}] training_step: After net.forward - yhat shape: {yhat.shape}, dtype: {yhat.dtype}", flush=True)
        print(f"[{dist.get_rank()}] training_step: yhat stats: min={yhat.min()}, max={yhat.max()}, mean={yhat.mean()}, std={yhat.std()}", flush=True)
        print(f"[{dist.get_rank()}] training_step: yhat NaN/Inf: NaN={torch.isnan(yhat).any()}, Inf={torch.isinf(yhat).any()}", flush=True)
    except Exception as e:
        print(f"[{dist.get_rank()}] CRITICAL ERROR in net.forward: {e}", flush=True)
        raise e

    try:
        yhat = clip_replace_constant(y, yhat, out_variables)
        print(f"[{dist.get_rank()}] training_step: After clip_replace_constant - yhat stats: min={yhat.min()}, max={yhat.max()}, mean={yhat.mean()}, std={yhat.std()}", flush=True)
        print(f"[{dist.get_rank()}] training_step: yhat (after clip) NaN/Inf: NaN={torch.isnan(yhat).any()}, Inf={torch.isinf(yhat).any()}", flush=True)
    except Exception as e:
        print(f"[{dist.get_rank()}] CRITICAL ERROR in clip_replace_constant: {e}", flush=True)
        raise e

    try:
        if y.size(dim=2) != yhat.size(dim=2) or y.size(dim=3) != yhat.size(dim=3):
            losses = train_loss_metric(
                yhat,
                y[:, :, 0 : yhat.size(dim=2), 0 : yhat.size(dim=3)],
                var_names=out_variables,
                var_weights=var_weights,
            )
        else:
            losses = train_loss_metric(
                yhat, y, var_names=out_variables, var_weights=var_weights
            )
        print(f"[{dist.get_rank()}] training_step: After train_loss_metric - losses shape: {losses.shape}, dtype: {losses.dtype}", flush=True)
        print(f"[{dist.get_rank()}] training_step: losses stats: min={losses.min()}, max={losses.max()}, mean={losses.mean()}, std={losses.std()}", flush=True)
        print(f"[{dist.get_rank()}] training_step: losses NaN/Inf: NaN={torch.isnan(losses).any()}, Inf={torch.isinf(losses).any()}", flush=True)
    except Exception as e:
        print(f"[{dist.get_rank()}] CRITICAL ERROR in train_loss_metric: {e}", flush=True)
        raise e

    loss_name = getattr(train_loss_metric, "name", "loss")
    if losses.dim() == 0:  # aggregate loss only
        loss = losses
    else:  # per channel + aggregate
        loss = losses[-1]

    print(f"[{dist.get_rank()}] training_step: Final loss value: {loss.item()}", flush=True)
    print(f"[{dist.get_rank()}] training_step: Final loss NaN/Inf: NaN={torch.isnan(loss).any()}, Inf={torch.isinf(loss).any()}", flush=True)

    return loss


def validation_step(
    batch, batch_idx: int, net, device: int, val_loss_metrics, val_target_transforms
) -> torch.Tensor:

    return evaluate_func(
        batch, "val", net, device, val_loss_metrics, val_target_transforms
    )


def evaluate_func(batch, stage: str, net, device: int, loss_metrics, target_transforms):

    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)

    yhat = net.forward(x, in_variables, out_variables)
    yhat = clip_replace_constant(y, yhat, out_variables)

    if stage == "val":
        loss_fns = loss_metrics
        transforms = target_transforms
    elif stage == "test":
        loss_fns = loss_metrics
        transforms = target_transforms
    else:
        raise RuntimeError("Invalid evaluation stage")
    loss_dict = {}
    for i, lf in enumerate(loss_fns):

        if transforms is not None and transforms[i] is not None:
            yhat_ = transforms[i](yhat)
            y_ = transforms[i](y)

        if y_.size(dim=2) != yhat_.size(dim=2) or y_.size(dim=3) != yhat_.size(dim=3):
            losses = lf(yhat_, y_[:, :, 0 : yhat_.size(dim=2), 0 : yhat_.size(dim=3)])
        else:
            losses = lf(yhat_, y_)

        loss_name = getattr(lf, "name", f"loss_{i}")
        if losses.dim() == 0:  # aggregate loss
            loss_dict[f"{stage}/{loss_name}:agggregate"] = losses
        else:  # per channel + aggregate
            for var_name, loss in zip(out_variables, losses):
                name = f"{stage}/{loss_name}:{var_name}"
                loss_dict[name] = loss
            loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
    return loss_dict


def setup_environment(local_rank):
    """
    Set up the distributed training environment.

    Args:
        local_rank (int): Local rank of the current process

    Returns:
        tuple: (device, world_size, world_rank, local_rank)
    """
    # Get distributed training information from SLURM environment
    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    # Set up device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Print process information
    print(f"Rank {world_rank}/{world_size} using device {device}", flush=True)

    return device, world_size, world_rank, local_rank


def create_model_and_optimizer(
    config, device, world_rank, data_module, in_vars, out_vars, train_loss_str
):
    """
    Create model, optimizer, and scheduler.

    Args:
        config (dict): Full configuration dictionary
        device: Training device
        world_rank (int): Process rank
        data_module: Data module for getting data dimensions
        in_vars (list): Input variable names
        out_vars (list): Output variable names
        train_loss_str (str): Loss function name

    Returns:
        tuple: (model, optimizer, scheduler, train_loss, val_losses)
    """
    # Extract model configuration
    preset = config["model"]["preset"]
    model_kwargs = config["model"].copy()
    model_kwargs.pop("preset", None)  # Remove preset from kwargs

    # Set up model
    (
        model,
        train_loss,
        val_losses,
        test_losses,
        train_transform,
        val_transforms,
        test_transforms,
    ) = cl.load_downscaling_module(
        device,
        model=None,
        data_module=data_module,
        architecture=preset,
        train_loss=train_loss_str,
        model_kwargs=model_kwargs,
    )

    if world_rank == 0:
        print(f"Created model: {preset}", flush=True)
        print(
            f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            flush=True,
        )

    # Extract optimizer configuration
    lr = float(config["model"]["lr"])
    weight_decay = float(config["model"]["weight_decay"])
    beta_1 = float(config["model"]["beta_1"])
    beta_2 = float(config["model"]["beta_2"])

    # Create optimizer
    optimizer = cl.load_optimizer(
        model,
        "adamw",
        {"lr": lr, "weight_decay": weight_decay, "betas": (beta_1, beta_2)},
    )

    # Extract scheduler configuration
    warmup_epochs = config["model"]["warmup_epochs"]
    warmup_start_lr = float(config["model"]["warmup_start_lr"])
    eta_min = float(config["model"]["eta_min"])
    max_epochs = config["trainer"]["max_epochs"]

    # Create scheduler
    scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {
            "warmup_epochs": warmup_epochs,
            "max_epochs": max_epochs,
            "warmup_start_lr": warmup_start_lr,
            "eta_min": eta_min,
        },
    )

    return model, optimizer, scheduler, train_loss, val_losses


def create_data_module(data_key, config, world_rank, device, do_tiling, div, overlap):
    """
    Create data module and loaders for training.

    Args:
        data_key (str): Dataset identifier (e.g., 'ERA5_1', 'PRISM')
        config (dict): Full configuration dictionary
        world_rank (int): Process rank
        device: Training device
        do_tiling (bool): Whether to use TILES algorithm
        div (int): Tile division factor
        overlap (int): Tile overlap in pixels

    Returns:
        tuple: (data_module, train_dataloader, val_dataloader, lat, lon)
    """
    # Extract configuration
    low_res_dir = config["data"]["low_res_dir"]
    high_res_dir = config["data"]["high_res_dir"]
    dict_in_variables = config["data"]["dict_in_variables"]
    dict_out_variables = config["data"]["dict_out_variables"]
    default_vars = config["data"]["default_vars"]

    batch_size = config["trainer"]["batch_size"]
    num_workers = config["trainer"]["num_workers"]
    buffer_size = config["trainer"]["buffer_size"]

    # Get variables for this dataset
    in_vars = dict_in_variables.get(data_key, default_vars)
    out_vars = dict_out_variables.get(data_key, ["2m_temperature"])

    if world_rank == 0:
        print(f"Creating data module for {data_key}", flush=True)
        print(f"Input variables: {in_vars}", flush=True)
        print(f"Output variables: {out_vars}", flush=True)
        log_gpu_memory(device, f"before data_module {data_key}")

    # Create data module
    data_module = cl.data.IterDataModule(
        "downscaling",
        low_res_dir[data_key],
        high_res_dir[data_key],
        in_vars,
        out_vars,
        subsample=1,
        buffer_size=buffer_size,
    )

    # Check tiling compatibility
    if do_tiling:
        patch_size = config["model"]["patch_size"]
        lat, lon = data_module.get_lat_lon()
        yout = len(lat) // div
        yinp = yout // 4 + overlap

        if yinp % patch_size != 0:
            if world_rank == 0:
                print(f"Tile height: {yinp}, patch_size {patch_size}", flush=True)
                print(
                    f"Overlap must be adjusted to accommodate patch_size. Need to increase by {yinp % patch_size}",
                    flush=True,
                )
            sys.exit("Please adjust overlap according to the instructions above")

    if world_rank == 0:
        log_gpu_memory(device, f"after data_module {data_key}")

    # Create data loaders
    train_dataloader = data_module.train_dataloader(
        batch_size,
        num_workers,
        shuffle=True,
        prefetch_factor=2,
        enable_tiling=do_tiling,
        num_tile=div * div if do_tiling else 1,
        tile_size=128,  # This seems to be hardcoded in original
    )

    val_dataloader = data_module.val_dataloader(
        batch_size, num_workers, shuffle=False, prefetch_factor=2
    )

    return data_module, train_dataloader, val_dataloader


def run_training_epochs(
    model,
    optimizer,
    scheduler,
    scaler,
    train_dataloader,
    epoch_start,
    epoch_end,
    data_type,
    var_weights,
    train_loss,
    device,
    world_rank,
    tensor_par_size,
    min_scale,
    cp_save_path,
    local_rank,
    use_qat=False,
    qat_start_epoch=0,
):
    """Run training loop for specified epoch range.
    
    Args:
        model: Model to train
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        scaler: GradScaler for mixed precision (None for float32, required for bfloat16)
        train_dataloader: DataLoader for training data
        epoch_start (int): Starting epoch number
        epoch_end (int): Ending epoch number (exclusive)
        data_type (str): Data type for training ('float32' or 'bfloat16')
        var_weights: Variable weights for loss calculation
        train_loss: Loss function
        device: Training device
        world_rank (int): Global rank of current process
        tensor_par_size (int): Tensor parallel size
        min_scale (float): Minimum scale value for GradScaler (None for float32)
        cp_save_path (str): Path for saving checkpoints
        local_rank (int): Local rank for current process
        use_qat (bool): Whether QAT is enabled
        qat_start_epoch (int): Epoch to activate QAT
        
    Returns:
        int: Final epoch number
    """
    if world_rank == 0:
        print("Entering run_training_epochs...", flush=True)

    for epoch in range(epoch_start, epoch_end):
        # Activate QAT at specified epoch
        if use_qat and epoch == qat_start_epoch:
            if world_rank == 0:
                print(f"\n{'='*80}", flush=True)
                print(f"ACTIVATING QAT AT EPOCH {epoch}", flush=True)
                print(f"{'='*80}", flush=True)
            from climate_learn.utils import qat_utils
            qat_utils.enable_qat_mode(model, enable=True)
            # Reduce learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= 0.1
                if world_rank == 0:
                    print(f"Reduced learning rate: {old_lr} → {param_group['lr']}", flush=True)
            if world_rank == 0:
                print(f"{'='*80}\n", flush=True)
        
        model.train()
        epoch_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        if world_rank == 0:
            print(f"Starting epoch {epoch}", flush=True)
            print("Getting first batch from train_dataloader...", flush=True)

        for batch_idx, batch in enumerate(train_dataloader):
            if world_rank == 0:
                print(f"Processing batch {batch_idx}...", flush=True)
                print("Before cuda synchronize", flush=True)
                torch.cuda.synchronize(device=device)
                print("After cuda synchronize", flush=True)
                tic1 = time.perf_counter()

            print(f"[{world_rank}] Before calling training_step", flush=True)
            loss = training_step(
                batch, batch_idx, model, device, var_weights, train_loss
            )
            print(f"[{world_rank}] After calling training_step", flush=True)
            epoch_loss += loss.detach()

            if world_rank < tensor_par_size:
                print(
                    f"epoch: {epoch}, batch_idx: {batch_idx}, "
                    f"world_rank: {world_rank}, loss: {loss}",
                    flush=True,
                )

            optimizer.zero_grad()

            try:
                if data_type == "float32":
                    loss.backward()
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer) # Unscale before clipping
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler._scale < min_scale:
                        scaler._scale = torch.tensor(min_scale).to(scaler._scale)
            except RuntimeError as e:
                if "NaN" in str(e) or "nan" in str(e):
                    print(f"[{world_rank}] WARNING: NaN detected during backward/optimization. Skipping batch {batch_idx}. Error: {e}", flush=True)
                    optimizer.zero_grad() # Clear gradients
                    continue # Skip to next batch
                else:
                    raise e # Re-raise other errors

            if world_rank == 0:
                log_gpu_memory(
                    device,
                    f"batch_idx {batch_idx} get_lr {scheduler.get_lr()} after optimizer step",
                    world_rank,
                )

            if world_rank == 0:
                torch.cuda.synchronize(device=device)
                tic4 = time.perf_counter()
                if batch_idx % 10 == 0:
                    print(
                        f"Batch {batch_idx}: {(tic4-tic1):0.4f} seconds",
                        flush=True,
                    )

        scheduler.step()

        if world_rank == 0:
            print(f"Epoch {epoch} completed. Loss: {epoch_loss.item()}", flush=True)

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            cp_save_path,
            world_rank,
            local_rank,
            tensor_par_size,
            device,
            use_qat=use_qat,
            qat_start_epoch=qat_start_epoch,
        )

    return epoch_end


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    cp_save_path,
    world_rank,
    local_rank,
    tensor_par_size,
    device,
    use_qat=False,
    qat_start_epoch=0,
):
    """
    Save model checkpoint to disk.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch (int): Current epoch number
        cp_save_path (str): Directory path for saving checkpoints
        world_rank (int): Current process rank
        local_rank (int): Local rank for the current process
        tensor_par_size (int): Size of tensor parallelism
        device: Current device
        use_qat (bool): Whether QAT is enabled
        qat_start_epoch (int): Epoch when QAT was activated
    """
    # Create checkpoint directory if needed (only on rank 0)
    if world_rank == 0:
        if not os.path.exists(cp_save_path):
            os.makedirs(cp_save_path)
            print(f"Created checkpoint directory: {cp_save_path}", flush=True)

    # Log memory before saving
    if world_rank == 0:
        log_gpu_memory(device, "Before torch.save", world_rank)

    # Get model, optimizer, and scheduler states
    model_states = model.state_dict()
    optimizer_states = optimizer.state_dict()
    scheduler_states = scheduler.state_dict()

    # Determine if QAT is currently active
    qat_active = use_qat and epoch >= qat_start_epoch

    # Save checkpoint only for ranks that are part of tensor parallelism
    if world_rank < tensor_par_size:
        file_name = get_checkpoint_filename(
            cp_save_path, epoch, world_rank, tensor_par_size
        )

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_states,
            "optimizer_state_dict": optimizer_states,
            "scheduler_state_dict": scheduler_states,
        }

        # Add quantization metadata if QAT is enabled
        # This metadata allows visualize.py to automatically detect QAT checkpoints
        # and convert them to INT8 using convert_qat_to_quantized()
        # Note: convert_to_int8_simple.py is now optional since metadata is saved during training
        if use_qat:
            from climate_learn.utils import qat_utils
            qat_status = qat_utils.check_qat_status(model)
            checkpoint_dict["quantization"] = {
                "enabled": True,
                "method": "qat",
                "precision": "int8",
                "qat_start_epoch": qat_start_epoch,
                "qat_active": qat_active,
                "has_fake_quant": qat_status.get("has_fake_quant", False),
                "num_fake_quant_modules": qat_status.get("num_fake_quant_modules", 0),
            }
            if world_rank == 0:
                print(f"Added quantization metadata: QAT active={qat_active}, "
                      f"FakeQuant modules={qat_status.get('num_fake_quant_modules', 0)}", flush=True)

        torch.save(checkpoint_dict, file_name)

        if world_rank == 0:
            print(f"Saved checkpoint to: {file_name}", flush=True)

    # Log memory after saving
    log_gpu_memory(device, "After torch.save", world_rank)

    # Synchronize all processes
    dist.barrier(device_ids=[local_rank])

    # Clean up to free memory
    del model_states
    del optimizer_states
    del scheduler_states


def parse_config(config_path, world_rank):
    """
    Parse configuration from YAML file.

    Args:
        config_path (str): Path to configuration YAML file
        world_rank (int): Current process rank for logging

    Returns:
        dict: Configuration dictionary with all parameters
    """
    if world_rank == 0:
        print(f"Loading config from: {config_path}", flush=True)

    # Load YAML configuration
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    # Extract trainer configuration
    trainer_conf = {
        "max_epochs": conf["trainer"]["max_epochs"],
        "checkpoint_path": conf["trainer"]["checkpoint"],
        "batch_size": conf["trainer"]["batch_size"],
        "num_workers": conf["trainer"]["num_workers"],
        "buffer_size": conf["trainer"]["buffer_size"],
        "data_type": conf["trainer"]["data_type"],
        "train_loss": conf["trainer"]["train_loss"],
        "pretrain_path": conf["trainer"]["pretrain"],
    }

    # Extract parallelism configuration
    parallelism_conf = {
        "fsdp_size": conf["parallelism"]["fsdp"],
        "simple_ddp_size": conf["parallelism"]["simple_ddp"],
        "tensor_par_size": conf["parallelism"]["tensor_par"],
        "seq_par_size": conf["parallelism"]["seq_par"],
    }

    # Extract tiling configuration with defaults
    try:
        do_tiling = conf["tiling"]["do_tiling"]
        tiling_conf = {
            "do_tiling": do_tiling,
            "div": conf["tiling"]["div"] if do_tiling else 1,
            "overlap": conf["tiling"]["overlap"] if do_tiling else 0,
        }
    except KeyError:
        tiling_conf = {"do_tiling": False, "div": 1, "overlap": 0}

    # Extract data configuration
    data_conf = {
        "low_res_dir": conf["data"]["low_res_dir"],
        "high_res_dir": conf["data"]["high_res_dir"],
        "default_vars": conf["data"]["default_vars"],
        "dict_in_variables": conf["data"]["dict_in_variables"],
        "dict_out_variables": conf["data"]["dict_out_variables"],
        "var_weights": conf["data"].get("var_weights", {}),
    }

    # Extract model configuration
    model_conf = conf["model"]

    return {
        "trainer": trainer_conf,
        "parallelism": parallelism_conf,
        "tiling": tiling_conf,
        "data": data_conf,
        "model": model_conf,
    }


def main(device):
    """
    Main training function for intermediate downscaling model.

    This function coordinates the entire training pipeline:
    1. Sets up distributed training environment
    2. Loads configuration from YAML file
    3. Initializes parallel process groups
    4. Creates data loaders for multiple datasets
    5. Builds and distributes the model across GPUs
    6. Runs training loop with checkpointing

    Args:
        device: Initial device (will be overridden based on local rank)
    """

    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    print(
        "world_size",
        world_size,
        "world_rank",
        world_rank,
        "local_rank",
        local_rank,
        flush=True,
    )

    config_path = sys.argv[1]

    if world_rank == 0:
        print("config_path", config_path, flush=True)

    conf = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    max_epochs = conf["trainer"]["max_epochs"]
    checkpoint_path = conf["trainer"]["checkpoint"]
    batch_size = conf["trainer"]["batch_size"]
    num_workers = conf["trainer"]["num_workers"]
    buffer_size = conf["trainer"]["buffer_size"]
    data_type = conf["trainer"]["data_type"]
    gpu_type = conf["trainer"]["gpu_type"]
    train_loss_str = conf["trainer"]["train_loss"]
    pretrain_path = conf["trainer"]["pretrain"]
    
    # QAT configuration
    use_qat = conf["trainer"].get("use_qat", False)
    qat_start_epoch = conf["trainer"].get("qat_start_epoch", 0)
    
    # Force float32 for QAT (FakeQuantize doesn't support bfloat16)
    if use_qat and data_type == "bfloat16":
        if world_rank == 0:
            print("WARNING: QAT requires float32. Overriding data_type from bfloat16 to float32", flush=True)
        data_type = "float32"

    # Validate data type early to fail fast with clear error message
    validate_data_type(data_type)

    fsdp_size = conf["parallelism"]["fsdp"]
    simple_ddp_size = conf["parallelism"]["simple_ddp"]
    tensor_par_size = conf["parallelism"]["tensor_par"]
    seq_par_size = conf["parallelism"]["seq_par"]

    try:
        do_tiling = conf["tiling"]["do_tiling"]
        if do_tiling:
            div = conf["tiling"]["div"]
            overlap = conf["tiling"]["overlap"]
        else:
            div = 1
            overlap = 0
    except Exception:
        do_tiling = False
        div = 1
        overlap = 0

    low_res_dir = conf["data"]["low_res_dir"]
    high_res_dir = conf["data"]["high_res_dir"]
    preset = conf["model"]["preset"]
    var_weights = conf["data"]["var_weights"]
    dict_out_variables = conf["data"]["dict_out_variables"]
    dict_in_variables = conf["data"]["dict_in_variables"]
    default_vars = conf["data"]["default_vars"]
    spatial_resolution = conf["data"]["spatial_resolution"]

    lr = float(conf["model"]["lr"])
    beta_1 = float(conf["model"]["beta_1"])
    beta_2 = float(conf["model"]["beta_2"])
    weight_decay = float(conf["model"]["weight_decay"])
    warmup_epochs = conf["model"]["warmup_epochs"]
    warmup_start_lr = float(conf["model"]["warmup_start_lr"])
    eta_min = float(conf["model"]["eta_min"])

    superres_mag = conf["model"]["superres_mag"]
    cnn_ratio = conf["model"]["cnn_ratio"]
    patch_size = conf["model"]["patch_size"]
    embed_dim = conf["model"]["embed_dim"]
    depth = conf["model"]["depth"]
    decoder_depth = conf["model"]["decoder_depth"]
    num_heads = conf["model"]["num_heads"]
    mlp_ratio = conf["model"]["mlp_ratio"]
    drop_path = conf["model"]["drop_path"]
    drop_rate = conf["model"]["drop_rate"]

    data_par_size = fsdp_size * simple_ddp_size

    if world_rank == 0:
        print("\n" + "=" * 80)
        print("Training Configuration Summary")
        print("=" * 80)
        print(f"Model: {preset}, Parameters: {embed_dim}d {depth}L {num_heads}H")
        print(f"Training: {max_epochs} epochs, batch_size={batch_size}, lr={lr}")
        print(f"Data type: {data_type}, Loss: {train_loss_str}")
        print(f"QAT: {'Enabled' if use_qat else 'Disabled'}, Start epoch: {qat_start_epoch if use_qat else 'N/A'}")
        print(f"Checkpoint: {checkpoint_path if checkpoint_path else 'None'}")
        print(f"Pretrain: {pretrain_path if pretrain_path else 'None'}")
        print("=" * 80 + "\n", flush=True)
        print(
            "data_par_size",
            data_par_size,
            "fsdp_size",
            fsdp_size,
            "simple_ddp_size",
            simple_ddp_size,
            "tensor_par_size",
            tensor_par_size,
            "seq_par_size",
            seq_par_size,
            "division",
            div,
            "overlap",
            overlap,
            flush=True,
        )

    # initialize parallelism groups
    (
        seq_par_group,
        data_par_group,
        tensor_par_group,
        data_seq_ort_group,
        fsdp_group,
        simple_ddp_group,
    ) = init_par_groups(
        data_par_size=data_par_size,
        tensor_par_size=tensor_par_size,
        seq_par_size=seq_par_size,
        fsdp_size=fsdp_size,
        simple_ddp_size=simple_ddp_size,
        num_heads=num_heads,
    )

    if gpu_type == "amd":
        if data_type == "bfloat16":
            # FusedAttn_option = FusedAttn.CK
            print("Forcing FusedAttn.DEFAULT to avoid NaNs in CK backend", flush=True)
            FusedAttn_option = FusedAttn.DEFAULT
        else:
            FusedAttn_option = FusedAttn.DEFAULT
    else:
        FusedAttn_option = FusedAttn.DEFAULT

    model_kwargs = {
        "default_vars": default_vars,
        "superres_mag": superres_mag,
        "cnn_ratio": cnn_ratio,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depth": depth,
        "decoder_depth": decoder_depth,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "drop_path": drop_path,
        "drop_rate": drop_rate,
        "tensor_par_size": tensor_par_size,
        "tensor_par_group": tensor_par_group,
        "FusedAttn_option": FusedAttn_option,
    }

    if world_rank == 0:
        print("model_kwargs", model_kwargs, flush=True)

    if preset != "vit" and preset != "res_slimvit":
        print("Only supports vit or residual slim vit training.", flush=True)
        sys.exit("Not vit or res_slimvit architecture")

    # if both checkpoint and pretrain are available, use checkpoint
    if checkpoint_path is not None and pretrain_path is not None:
        pretrain_path = None

    model = None

    first_time_bool = True

    interval_epochs = 1

    epoch_start = 0

    cp_save_path = "checkpoints/climate"

    if data_type == "bfloat16":
        scaler = ShardedGradScaler(init_scale=8192, growth_interval=100)
        min_scale = 128
        if world_rank == 0:
            print("initialize ShardedGradScaler for bfloat16", flush=True)
    else:
        scaler = None
        min_scale = None

    while (epoch_start + interval_epochs) < max_epochs:

        for data_key in low_res_dir.keys():
            # Set up data

            in_vars = dict_in_variables[data_key]
            out_vars = dict_out_variables[data_key]

            if world_rank == 0:
                print("***************************", flush=True)
                print("data_key is ", data_key, flush=True)
                print("in_vars", in_vars, flush=True)
                print("out_vars", out_vars, flush=True)
                print("default_vars", default_vars, flush=True)
                log_gpu_memory(device, "before data_module")

            # load data module
            data_module = cl.data.IterDataModule(
                "downscaling",
                low_res_dir[data_key],
                high_res_dir[data_key],
                in_vars,
                out_vars=out_vars,
                data_par_size=data_par_size,
                data_par_group=data_par_group,
                subsample=1,
                batch_size=batch_size,
                buffer_size=buffer_size,
                num_workers=num_workers,
                div=div,
                overlap=overlap,
            ).to(device)

            data_module.setup()

            if do_tiling:
                lat, lon = data_module.get_lat_lon()
                yout = len(lat) // div
                yinp = yout // 4 + overlap
                if yinp % patch_size != 0:
                    if world_rank == 0:
                        print(f"Tile height: {yinp}, patch_size {patch_size}")
                        print(
                            "Overlap must be adjusted to accommodate patch_size of the "
                            "Transformer. Need to increase the overlap by ",
                            (yinp % patch_size),
                        )
                        sys.exit(
                            "Please increase the overlap accordingly to the instructions "
                            "in the print message"
                        )

            if world_rank == 0:
                log_gpu_memory(device, "after data_module")

            if first_time_bool:
                # Set up deep learning model
                (
                    model,
                    train_loss,
                    val_losses,
                    test_losses,
                    train_transform,
                    val_transforms,
                    test_transforms,
                ) = cl.load_downscaling_module(
                    device,
                    model=model,
                    data_module=data_module,
                    architecture=preset,
                    train_loss=train_loss_str,
                    model_kwargs=model_kwargs,
                )

                # Model and loss functions created successfully

                model = model.to(device)

                if torch.distributed.get_rank() == 0:
                    print("Loading model weights...", flush=True)
                    # Print only model summary instead of all parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    print(
                        f"Total model parameters: {total_params / 1e6:.2f}M", flush=True
                    )

                # load from checkpoint for continued training , or from pretrained model weights
                load_checkpoint_pretrain(
                    model,
                    checkpoint_path,
                    pretrain_path,
                    cp_save_path,
                    tensor_par_size=tensor_par_size,
                    tensor_par_group=tensor_par_group,
                )
                
                # Prepare model for QAT if enabled (before FSDP wrapping)
                # Note: QAT preparation must happen before FSDP wrapping because
                # prepare_qat modifies the model structure by inserting FakeQuantize modules
                if use_qat:
                    if world_rank == 0:
                        print("\nPreparing model for QAT (before FSDP wrapping)...", flush=True)
                    from climate_learn.utils import qat_utils
                    model = qat_utils.prepare_model_for_qat(
                        model, 
                        attention_only=True,
                        tensor_par_size=tensor_par_size,
                    )
                    # Start with QAT disabled (will enable at qat_start_epoch)
                    qat_utils.enable_qat_mode(model, enable=False)
                    if world_rank == 0:
                        print(f"✓ QAT prepared. Will activate at epoch {qat_start_epoch}", flush=True)

                # Model weights loaded, no need to print all parameters again

                seed_everything(0)

                # set up layer wrapping
                if preset == "vit" or preset == "res_slimvit":

                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls={
                            Block,
                            Sequential,  # < ---- Your Transformer layer class
                        },
                    )

                    check_fn = lambda submodule: isinstance(
                        submodule, Block
                    ) or isinstance(submodule, Sequential)

                if data_type == "float32":
                    precision_dt = torch.float32
                elif data_type == "bfloat16":
                    precision_dt = torch.bfloat16
                else:
                    raise RuntimeError("Data type not supported")

                # floating point policy
                bfloatPolicy = MixedPrecision(
                    param_dtype=precision_dt,
                    # Gradient communication precision.
                    reduce_dtype=precision_dt,
                    # Buffer precision.
                    buffer_dtype=precision_dt,
                )

                # hybrid sharded FSDP
                if fsdp_size > 1 and simple_ddp_size > 1:

                    print("enter hybrid FSDP", flush=True)
                    model = FSDP(
                        model,
                        device_id=local_rank,
                        process_group=(fsdp_group, simple_ddp_group),
                        sync_module_states=True,
                        sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=bfloatPolicy,
                        forward_prefetch=True,
                        limit_all_gathers=False,
                    )
                # fully sharded FSDP
                elif fsdp_size > 1 and simple_ddp_size == 1:
                    print("enter fully sharded FSDP", flush=True)
                    model = FSDP(
                        model,
                        device_id=local_rank,
                        process_group=fsdp_group,
                        sync_module_states=True,
                        sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=bfloatPolicy,
                        forward_prefetch=True,
                        limit_all_gathers=False,
                    )
                else:
                    # no shard only
                    print("enter NO SHARD only,", flush=True)
                    model = FSDP(
                        model,
                        device_id=local_rank,
                        process_group=simple_ddp_group,
                        sync_module_states=True,
                        sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=bfloatPolicy,
                        forward_prefetch=True,
                        limit_all_gathers=False,
                    )

            # Verify QAT status after FSDP wrapping (for logging/debugging)
            if use_qat:
                if world_rank == 0:
                    print("\nVerifying QAT status after FSDP wrapping...", flush=True)
                from climate_learn.utils import qat_utils
                qat_status = qat_utils.check_qat_status(model)
                if world_rank == 0:
                    print(f"QAT Status: has_fake_quant={qat_status['has_fake_quant']}, "
                          f"num_modules={qat_status['num_fake_quant_modules']}", flush=True)
                    if not qat_status['has_fake_quant']:
                        print("WARNING: No FakeQuantize modules found after FSDP wrapping!", flush=True)
                    print("="*80 + "\n", flush=True)

            # Update spatial resolution, image size, and number of variables to model
            # based on datasets
            in_shape, _ = data_module.get_data_dims()
            _, in_height, in_width = in_shape[1:]

            with FSDP.summon_full_params(model):
                model.data_config(
                    spatial_resolution[data_key],
                    (in_height, in_width),
                    len(in_vars),
                    len(out_vars),
                )

            if first_time_bool:
                # activation checkpointing
                apply_activation_checkpointing(
                    model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
                )

                # load optimzier and scheduler

                optimizer = cl.load_optimizer(
                    model,
                    "adamw",
                    {"lr": lr, "weight_decay": weight_decay, "betas": (beta_1, beta_2)},
                )

                scheduler = cl.load_lr_scheduler(
                    "linear-warmup-cosine-annealing",
                    optimizer,
                    {
                        "warmup_epochs": warmup_epochs,
                        "max_epochs": max_epochs,
                        "warmup_start_lr": warmup_start_lr,
                        "eta_min": eta_min,
                    },
                )

                if checkpoint_path is not None:

                    print(
                        "optimizer resume from checkpoint",
                        checkpoint_path,
                        " Checkpoint path found.",
                        flush=True,
                    )
                    src_rank = world_rank - tensor_par_size * dist.get_rank(
                        group=data_seq_ort_group
                    )
                    map_location = "cpu"
                    checkpoint_path = get_tensor_parallel_checkpoint_path(
                        checkpoint_path, src_rank, tensor_par_size
                    )

                    checkpoint = torch.load(checkpoint_path, map_location=map_location)
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    epoch_start = checkpoint["epoch"] + 1
                    del checkpoint

            # get latitude and longitude
            lat, lon = data_module.get_lat_lon()

            # get train data loader
            train_dataloader = data_module.train_dataloader()

            # get validation data loader
            val_dataloader = data_module.val_dataloader()

            # perform training

            epoch_end = epoch_start + interval_epochs
            epoch_end = epoch_end if epoch_end < max_epochs else max_epochs

            epoch_start = run_training_epochs(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_dataloader=train_dataloader,
                epoch_start=epoch_start,
                epoch_end=epoch_end,
                data_type=data_type,
                var_weights=var_weights,
                train_loss=train_loss,
                device=device,
                world_rank=world_rank,
                tensor_par_size=tensor_par_size,
                min_scale=min_scale,
                cp_save_path=cp_save_path,
                local_rank=local_rank,
                use_qat=use_qat,
                qat_start_epoch=qat_start_epoch,
            )

            if first_time_bool:
                first_time_bool = False


if __name__ == "__main__":
    # Check if SLURM environment variables are set
    if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
        os.environ["MASTER_PORT"] = "29500"
        # os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"] # Already set by SLURM
        # os.environ["RANK"] = os.environ["SLURM_PROCID"] # Already set by SLURM

        world_size = int(os.environ["SLURM_NTASKS"])
        world_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        # Default to single process for local development/testing
        print("SLURM environment variables not found. Defaulting to single-process (rank 0 of 1).", flush=True)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500" # Use a fixed port
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0" # Custom env var for local_rank

        world_size = 1
        world_rank = 0
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=7200000),
        rank=world_rank,
        world_size=world_size,
    )
    torch.autograd.set_detect_anomaly(True)

    print("Using dist.init_process_group. world_size ", world_size, flush=True)

    main(device)

    dist.destroy_process_group()
