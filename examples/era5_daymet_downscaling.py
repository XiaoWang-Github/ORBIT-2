# Standard library
from argparse import ArgumentParser
import os
import torch
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
from torch.cuda.amp.grad_scaler import GradScaler
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
import numpy as np

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    SR_PRESSURE_LEVELS,
    CONSTANTS,
)
from timm.models.vision_transformer import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock,
)

# Import common utilities
from utils import seed_everything, log_gpu_memory


def load_checkpoint(model, checkpoint_path, device, world_rank=0):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        sys.exit(f"Checkpoint path does not exist: {checkpoint_path}")

    if world_rank == 0:
        print(f"Loading checkpoint from: {checkpoint_path}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint

    if world_rank == 0:
        print("Checkpoint loaded successfully", flush=True)


def load_training_state(optimizer, scheduler, checkpoint_path, world_rank=0):
    """Load optimizer and scheduler state from checkpoint."""
    if world_rank == 0:
        print(f"Loading training state from checkpoint: {checkpoint_path}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch_start = checkpoint["epoch"] + 1
    del checkpoint

    if world_rank == 0:
        print(f"Resuming from epoch {epoch_start}", flush=True)

    return epoch_start


def load_pretrained_weights(model, pretrained_path, device, world_rank=0):
    """Load pretrained weights into model."""
    checkpoint = torch.load(pretrained_path, map_location="cpu")

    if world_rank == 0:
        print(f"Loading pre-trained checkpoint from: {pretrained_path}")

    pretrain_model = checkpoint["model_state_dict"]
    del checkpoint

    state_dict = model.state_dict()

    if world_rank == 0:
        for k in list(pretrain_model.keys()):
            print(f"Pretrained model before deletion. Name: {k}", flush=True)

    # Remove or rename specific keys
    for k in list(pretrain_model.keys()):
        if "pos_embed" in k:
            if world_rank == 0:
                print(f"Removing pos_embed")
            del pretrain_model[k]
        if "var_" in k:
            if world_rank == 0:
                print(f"Removing var_embed, var_query and var_agg")
            del pretrain_model[k]
        if "token_embeds" in k:
            if world_rank == 0:
                print(f"Removing token_embed")
            del pretrain_model[k]
        if "channel" in k:
            if world_rank == 0:
                print(f"Renaming key: {k}")
            pretrain_model[k.replace("channel", "var")] = pretrain_model[k]
            del pretrain_model[k]
    # Remove keys that don't exist or have mismatched shapes
    for k in list(pretrain_model.keys()):
        if k not in state_dict.keys():
            if world_rank == 0:
                print(f"Removing key {k}: not in model")
            del pretrain_model[k]
        elif pretrain_model[k].shape != state_dict[k].shape:
            if world_rank == 0:
                print(
                    f"Removing key {k}: shape mismatch {pretrain_model[k].shape} vs {state_dict[k].shape}"
                )
            del pretrain_model[k]

    if world_rank == 0:
        print(f"Loading {len(pretrain_model)} keys into model")

    # load pre-trained model
    msg = model.load_state_dict(pretrain_model, strict=False)
    if world_rank == 0:
        print(f"Load state dict result: {msg}")
    del pretrain_model


def replace_constant(y, yhat, out_variables):
    """Replace predicted constants with ground truth values."""
    for i, var in enumerate(out_variables):
        if var in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat


def training_step(
    batch, batch_idx, net, device: int, train_loss_metric, train_target_transform
) -> torch.Tensor:
    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
    # NOTE debug memo: here check distibution of output precip if log-transformed or not
    # np.save("./test_y_LogTF.npy", y.detach().cpu().numpy())

    yhat = net.forward(x)
    yhat = replace_constant(y, yhat, out_variables)
    losses = train_loss_metric(yhat, y)
    loss_name = getattr(train_loss_metric, "name", "loss")
    if losses.dim() == 0:  # aggregate loss only
        loss = losses
    else:  # per channel + aggregate
        loss = losses[-1]

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

    yhat = net.forward(x)
    yhat = replace_constant(y, yhat, out_variables)

    if stage == "val":
        loss_fns = loss_metrics
        transforms = target_transforms
    elif stage == "test":
        loss_fns = loss_metrics
        transforms = self.target_transforms
    else:
        raise RuntimeError("Invalid evaluation stage")
    loss_dict = {}
    for i, lf in enumerate(loss_fns):

        if transforms is not None and transforms[i] is not None:
            yhat_ = transforms[i](yhat)
            y_ = transforms[i](y)
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


# Removed - using seed_everything from utils.py


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser(description="ERA5-DAYMET Downscaling with Climate Models")
    parser.add_argument("era5_daymet_low_res_dir", help="Path to low resolution data")
    parser.add_argument("era5_daymet_high_res_dir", help="Path to high resolution data")
    parser.add_argument(
        "preset",
        choices=["resnet", "unet", "vit", "res_slimvit"],
        help="Model architecture",
    )
    parser.add_argument(
        "variable",
        choices=["t2m", "z500", "t850", "u10", "tp", "prcp"],
        help="The variable to predict",
    )
    parser.add_argument("--summary_depth", type=int, default=1)
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--loss_function",
        choices=["mse", "perceptual", "quantile", "imagegradient", "masked_mse"],
        default="mse",
        help="Loss function",
    )
    parser.add_argument("--pretrain", default=None, help="Pretrained model path")
    return parser.parse_args()


def setup_distributed():
    """Setup distributed training environment."""
    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    if world_rank == 0:
        print(
            f"Distributed setup: world_size={world_size}, world_rank={world_rank}, "
            f"local_rank={local_rank}",
            flush=True,
        )

    return world_size, world_rank, local_rank


def setup_fsdp_policies(preset):
    """Setup FSDP wrapping policies based on model architecture."""
    if preset in ["vit", "res_slimvit"]:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Block, Sequential}
        )
        check_fn = lambda submodule: isinstance(submodule, (Block, Sequential))
    elif preset == "unet":
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={UpBlock, DownBlock, MiddleBlock},
        )
        check_fn = lambda submodule: isinstance(
            submodule, (UpBlock, DownBlock, MiddleBlock)
        )
    else:  # resnet
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={ResidualBlock}
        )
        check_fn = lambda submodule: isinstance(submodule, ResidualBlock)

    return auto_wrap_policy, check_fn


def save_checkpoint(model, optimizer, scheduler, epoch, root_dir, world_rank, device):
    """Save training checkpoint."""
    checkpoint_path = f"{root_dir}/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)

    log_gpu_memory(device, "Before saving checkpoint", world_rank)

    model_states = model.state_dict()
    optimizer_states = optimizer.state_dict()
    scheduler_states = scheduler.state_dict()

    if world_rank == 0:
        filename = f"{checkpoint_path}/ERA5-Daymet_epoch_{epoch}.ckpt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_states,
                "optimizer_state_dict": optimizer_states,
                "scheduler_state_dict": scheduler_states,
            },
            filename,
        )
        print(f"Checkpoint saved: {filename}", flush=True)

    log_gpu_memory(device, "After saving checkpoint", world_rank)

    dist.barrier()
    del model_states
    del optimizer_states
    del scheduler_states


def run_validation(
    model, val_dataloader, epoch, device, val_losses, val_transforms, world_rank
):
    """Run validation phase."""
    model.eval()

    if world_rank == 0:
        print(f"\nValidation - Epoch {epoch}", flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if world_rank == 0:
                torch.cuda.synchronize(device=device)
                tic1 = time.perf_counter()

            losses = validation_step(
                batch, batch_idx, model, device, val_losses, val_transforms
            )

            if world_rank == 0:
                print(
                    f"Val epoch: {epoch}, batch_idx: {batch_idx}, " f"losses: {losses}",
                    flush=True,
                )

                torch.cuda.synchronize(device=device)
                tic4 = time.perf_counter()
                print(
                    f"Validation batch time: {(tic4-tic1):0.4f} seconds\n", flush=True
                )


def setup_input_variables(target_variable, world_rank=0):
    """Setup input variables based on target variable."""
    variables = [
        "land_sea_mask",
        "orography",
        "lattitude",
        "sea_surface_temperature",
        "geopotential",
        "temperature",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
    ]

    out_var_dict = {"prcp": "prcp"}  # daymet derived

    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in SR_PRESSURE_LEVELS:
                in_vars.append(f"{var}_{level}")
        else:
            in_vars.append(var)

    if world_rank == 0:
        print(f"Input variables: {in_vars}", flush=True)

    return in_vars, out_var_dict[target_variable]


def main(device):
    """Main training function."""
    # Setup distributed training
    world_size, world_rank, local_rank = setup_distributed()

    # Parse command line arguments
    args = parse_arguments()

    if world_rank == 0:
        print(f"Arguments: {args}", flush=True)

    # Checkpoint takes priority over pretrain
    if args.checkpoint is not None and args.pretrain is not None:
        args.pretrain = None
        if world_rank == 0:
            print("Using checkpoint instead of pretrain", flush=True)

    # Setup input variables
    in_vars, out_var = setup_input_variables(args.variable, world_rank)

    # Create data module
    data_module = cl.data.IterDataModule(
        "downscaling",
        args.era5_daymet_low_res_dir,
        args.era5_daymet_high_res_dir,
        in_vars,
        out_vars=[out_var],
        subsample=1,
        batch_size=args.batch_size,
        buffer_size=400,
        num_workers=1,
    ).to(device)
    data_module.setup()

    # Create model
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
        data_module=data_module,
        architecture=args.preset,
        train_loss=args.loss_function,
        train_target_transform=None,
    )

    if world_rank == 0:
        print(
            f"Model setup: train_loss={train_loss}, train_transform={train_transform}",
            flush=True,
        )

    model = model.to(device)

    # Load checkpoint or pretrained weights
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, device, world_rank)
    elif args.pretrain is not None:
        load_pretrained_weights(model, args.pretrain, device, world_rank)

    # Set random seed
    seed_everything(0)

    # Set default root directory
    default_root_dir = f"{args.preset}_downscaling_{args.variable}/{args.loss_function}"

    # Setup layer wrapping policy based on architecture
    auto_wrap_policy, check_fn = setup_fsdp_policies(args.preset)

    # Setup mixed precision policy
    bfloat_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        device_id=local_rank,
        process_group=None,
        sync_module_states=True,
        sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bfloat_policy,
        forward_prefetch=True,
        limit_all_gathers=False,
    )

    # Apply activation checkpointing
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )

    if world_rank == 0:
        print(f"Model architecture:\n{model}", flush=True)

    # Setup optimizer and scheduler
    optimizer = cl.load_optimizer(
        model, "adamw", {"lr": 5e-5, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    )

    scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {
            "warmup_epochs": 2,
            "max_epochs": args.max_epochs,
            "warmup_start_lr": 1e-7,
            "eta_min": 1e-7,
        },
    )

    # Resume from checkpoint if specified
    epoch_start = 0
    if args.checkpoint is not None:
        epoch_start = load_training_state(
            optimizer, scheduler, args.checkpoint, world_rank
        )

    # Get data loaders
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # Training loop
    for epoch in range(epoch_start, args.max_epochs):
        model.train()
        epoch_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        if world_rank == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{args.max_epochs-1}")
            print(f"{'='*50}", flush=True)

        # Training phase
        for batch_idx, batch in enumerate(train_dataloader):
            if world_rank == 0:
                torch.cuda.synchronize(device=device)
                tic1 = time.perf_counter()

            loss = training_step(
                batch, batch_idx, model, device, train_loss, train_target_transform=None
            )
            # exit(0)

            epoch_loss += loss.detach()

            if world_rank == 0:
                print(
                    "epoch: ",
                    epoch,
                    "batch_idx",
                    batch_idx,
                    "world_rank",
                    world_rank,
                    " loss ",
                    loss,
                    flush=True,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if world_rank == 0:
                if batch_idx % 10 == 0:
                    print(
                        "rank",
                        world_rank,
                        "batch_idx",
                        batch_idx,
                        "get_lr ",
                        scheduler.get_lr(),
                        "after optimizer step torch.cuda.memory_reserved: %fGB"
                        % (torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024),
                        flush=True,
                    )

            if world_rank == 0:
                torch.cuda.synchronize(device=device)
                tic4 = time.perf_counter()
                print(
                    f"my rank {dist.get_rank()}. tic4-tic1 in {(tic4-tic1):0.4f} seconds\n",
                    flush=True,
                )

        scheduler.step()

        if world_rank == 0:
            epoch_loss = epoch_loss / (batch_idx + 1)
            print("epoch: ", epoch, " epoch_loss ", epoch_loss, flush=True)

        if world_rank == 0:
            print(f"Epoch {epoch} loss: {epoch_loss.item():.6f}", flush=True)

        # Save checkpoint periodically
        if epoch % 5 == 0 or epoch == args.max_epochs - 1:
            save_checkpoint(
                model, optimizer, scheduler, epoch, default_root_dir, world_rank, device
            )

        # Validation phase
        if epoch % 2 == 0:
            run_validation(
                model,
                val_dataloader,
                epoch,
                device,
                val_losses,
                val_transforms,
                world_rank,
            )


if __name__ == "__main__":
    # Setup distributed environment
    os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["RANK"] = os.environ["SLURM_PROCID"]

    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    # Setup device
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # Initialize distributed training
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=7200000),
        rank=world_rank,
        world_size=world_size,
    )

    if world_rank == 0:
        print(f"Distributed training initialized. World size: {world_size}", flush=True)

    # Run main training
    main(device)

    # Clean up
    dist.destroy_process_group()
