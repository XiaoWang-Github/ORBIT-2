"""Visualization script for ORBIT-2 downscaling model outputs.

This script loads a pre-trained ORBIT-2 model and generates visualizations
of downscaled climate data. It supports distributed execution across multiple
GPUs using FSDP (Fully Sharded Data Parallel) and tensor parallelism.

Usage:
    python visualize.py config.yaml [options]

Example:
    python visualize.py ../configs/interm_8m_ft.yaml --index 0 --variable total_precipitation_24hr
"""

import climate_learn as cl
import torch
import os
import functools
from argparse import ArgumentParser
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
from datetime import timedelta
import sys
import time
import yaml

from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
)
from climate_learn.models.hub.components.vit_blocks import Block
from torch.nn import Sequential
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from climate_learn.utils.fused_attn import FusedAttn
from utils import seed_everything, init_par_groups


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


def load_pretrained_weights(
    model, pretrain_path, device, tensor_par_size=1, tensor_par_group=None
):
    """Load pretrained model weights for visualization.

    This function loads only the model weights without optimizer or scheduler states,
    which is sufficient for inference/visualization tasks. It handles tensor parallel
    models by loading the appropriate rank-specific checkpoint.

    Args:
        model: PyTorch model to load weights into
        pretrain_path (str): Path to the pretrained model checkpoint
        device: Device to load the model on (e.g., torch.device('cuda:0'))
        tensor_par_size (int): Size of tensor parallelism (default: 1)
        tensor_par_group: Process group for tensor parallelism (default: None)
    """
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    # Adjust path for tensor parallel checkpoints
    if tensor_par_size > 1 and pretrain_path is not None:
        pretrain_path = pretrain_path + "_" + "rank" + "_" + str(world_rank)

    print("world_rank", world_rank, "pretrain_path", pretrain_path, flush=True)

    # load pretrained model
    if world_rank < tensor_par_size:
        if pretrain_path is None:
            print(
                "world_rank",
                world_rank,
                "No pretrained model path provided in config.",
                flush=True,
            )
            sys.exit("pretrain_path is None - please specify pretrain path in config file")
        elif os.path.exists(pretrain_path):
            print(
                "world_rank",
                world_rank,
                "load pretrained model",
                pretrain_path,
                " Pretrain path found.",
                flush=True,
            )
            _load_pretrained_weights(model, pretrain_path, device, world_rank)
        else:
            print(
                "resume from pretrained model was set to True. But the pretrained model path does not exist.",
                flush=True,
            )
            sys.exit("pretrain path does not exist")

    dist.barrier(device_ids=[local_rank])


def _load_pretrained_weights(model, pretrain_path, device, world_rank):
    """Internal function to load and process pretrained weights.

    Args:
        model: Target model to load weights into
        pretrain_path (str): Path to checkpoint file
        device: Device to load the model on
        world_rank (int): Global rank of the current process
    """
    # Load to CPU first to avoid GPU memory issues
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

    # Remove keys that don't exist in the target model or have shape mismatches
    for k in list(pretrain_model.keys()):  # Iterate through pretrained model keys
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

    # Load pre-trained model
    msg = model.load_state_dict(pretrain_model, strict=False)
    print(msg)
    del pretrain_model


def main():
    """Main function for model visualization.

    This function orchestrates the entire visualization pipeline:
    1. Sets up distributed training environment
    2. Loads configuration from YAML file
    3. Initializes data modules and model
    4. Loads pretrained weights
    5. Runs visualization on specified data samples
    """
    # Parse command line arguments first
    parser = ArgumentParser(description="Visualize ORBIT-2 model outputs")
    parser.add_argument("config", type=str, help="Path to configuration YAML file")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of test sample to visualize (default: 0)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="total_precipitation_24hr",
        help="Variable to visualize (default: total_precipitation_24hr)",
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default="29500",
        help="Master port for distributed training (default: 29500)",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["float32", "bfloat16"],
        default=None,
        help="Override data type from config (default: use config value)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint file (.ckpt). If provided, overrides the 'pretrain' path in config file",
    )
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["RANK"] = os.environ["SLURM_PROCID"]

    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    torch.distributed.init_process_group(
        "nccl",
        timeout=timedelta(seconds=7200000),
        rank=world_rank,
        world_size=world_size,
    )

    config_path = args.config

    if world_rank == 0:
        print("config_path", config_path, flush=True)

    conf = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    max_epochs = conf["trainer"]["max_epochs"]
    checkpoint_path = conf["trainer"]["checkpoint"]
    batch_size = conf["trainer"]["batch_size"]
    num_workers = conf["trainer"]["num_workers"]
    buffer_size = conf["trainer"]["buffer_size"]
    spatial_resolution = conf["data"]["spatial_resolution"]
    
    # Priority: 1) Command line argument, 2) Config pretrain path
    if args.checkpoint:
        pretrain_path = args.checkpoint
        if world_rank == 0:
            print(f"Using checkpoint from command line: {pretrain_path}", flush=True)
    else:
        pretrain_path = conf["trainer"]["pretrain"]
        if world_rank == 0:
            print(f"Using checkpoint from config: {pretrain_path}", flush=True)
    # Use command line override if provided, otherwise use config value
    # Force float32 for visualization to avoid numpy conversion issues with bfloat16
    data_type = args.data_type or "float32"
    
    # Validate data type
    validate_data_type(data_type)
    
    if world_rank == 0 and conf["trainer"].get("data_type") == "bfloat16":
        print("Note: Forcing float32 for visualization (bfloat16 not supported for numpy conversion)", flush=True)

    # Load tiling configuration for TILES algorithm
    try:
        do_tiling = conf["tiling"]["do_tiling"]
        if do_tiling:
            div = conf["tiling"]["div"]  # Number of divisions per dimension
            overlap = conf["tiling"]["overlap"]  # Overlap between tiles in pixels
        else:
            div = 1
            overlap = 0
    except Exception:
        print("Tiling parameter not found. Using default: no tiling", flush=True)
        do_tiling = False
        div = 1
        overlap = 0

    tensor_par_size = conf["parallelism"]["tensor_par"]
    fsdp_size = world_size // tensor_par_size
    simple_ddp_size = 1
    seq_par_size = 1

    FusedAttn_option = FusedAttn.DEFAULT

    low_res_dir = conf["data"]["low_res_dir"]
    high_res_dir = conf["data"]["high_res_dir"]
    preset = conf["model"]["preset"]
    dict_out_variables = conf["data"]["dict_out_variables"]
    dict_in_variables = conf["data"]["dict_in_variables"]
    default_vars = conf["data"]["default_vars"]

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
        print(
            "max_epochs",
            max_epochs,
            " ",
            checkpoint_path,
            " ",
            pretrain_path,
            " ",
            low_res_dir,
            " ",
            high_res_dir,
            "preset",
            preset,
            "dict_out_variables",
            dict_out_variables,
            "lr",
            lr,
            "beta_1",
            beta_1,
            "beta_2",
            beta_2,
            "weight_decay",
            weight_decay,
            "warmup_epochs",
            warmup_epochs,
            "warmup_start_lr",
            warmup_start_lr,
            "eta_min",
            eta_min,
            "superres_mag",
            superres_mag,
            "cnn_ratio",
            cnn_ratio,
            "patch_size",
            patch_size,
            "embed_dim",
            embed_dim,
            "depth",
            depth,
            "decoder_depth",
            decoder_depth,
            "num_heads",
            num_heads,
            "mlp_ratio",
            mlp_ratio,
            "drop_path",
            drop_path,
            "drop_rate",
            drop_rate,
            "batch_size",
            batch_size,
            "num_workers",
            num_workers,
            "buffer_size",
            buffer_size,
            flush=True,
        )
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
            "FusedAttn_option",
            FusedAttn_option,
            "div",
            div,
            "overlap",
            overlap,
            flush=True,
        )

    # Initialize distributed parallel process groups for model training
    # Returns: seq_par_group, data_par_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group
    _, data_par_group, tensor_par_group, _, fsdp_group, _ = init_par_groups(
        data_par_size=data_par_size,
        tensor_par_size=tensor_par_size,
        seq_par_size=seq_par_size,
        fsdp_size=fsdp_size,
        simple_ddp_size=simple_ddp_size,
        num_heads=num_heads,
    )

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

    # Set up data
    # Automatically determine which dataset to use based on config
    data_key = None
    for key in dict_in_variables.keys():
        if key in low_res_dir and key in high_res_dir:
            data_key = key
            break

    if data_key is None:
        print("No matching dataset found in config. Using first available key.")
        data_key = list(dict_in_variables.keys())[0]

    in_vars = dict_in_variables[data_key]
    out_vars = dict_out_variables[data_key]

    if world_rank == 0:
        print("in_vars", in_vars, flush=True)
        print("out_vars", out_vars, flush=True)

    # Initialize data module for training data with tiling support
    data_module = cl.data.IterDataModule(
        "downscaling",
        low_res_dir[data_key],
        high_res_dir[data_key],
        in_vars,
        out_vars=out_vars,
        data_par_size=data_par_size,
        data_par_group=data_par_group,
        subsample=1,
        batch_size=1,
        buffer_size=buffer_size,
        num_workers=num_workers,
        div=div,
        overlap=overlap,
    ).to(device)

    data_module.setup()

    # Initialize separate data module for visualization without tiling
    # This ensures we visualize the full image, not individual tiles
    dm_vis = cl.data.IterDataModule(
        "downscaling",
        low_res_dir[data_key],
        high_res_dir[data_key],
        in_vars,
        out_vars=out_vars,
        data_par_size=data_par_size,
        data_par_group=data_par_group,
        subsample=1,
        batch_size=1,
        buffer_size=buffer_size,
        num_workers=num_workers,
        div=1,
        overlap=0,
    ).to(device)

    dm_vis.setup()

    # Initialize the ORBIT-2 model with specified architecture and parameters
    (
        model,
        train_loss,
        val_losses,
        test_losses,
        train_transform,
        val_transforms,
        test_transforms,
    ) = cl.load_downscaling_module(
        device, data_module=data_module, architecture=preset, model_kwargs=model_kwargs
    )

    if dist.get_rank() == 0:
        print(
            "train_loss",
            train_loss,
            "train_transform",
            train_transform,
            "img_size",
            model.img_size,
            flush=True,
        )

    model = model.to(device)

    # Get denormalization transform for converting model outputs back to physical units
    denorm = test_transforms[0]

    print("denorm is ", denorm, flush=True)

    # Load pretrained model weights from checkpoint
    load_pretrained_weights(
        model,
        pretrain_path,
        device,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
    )

    if torch.distributed.get_rank() == 0:
        print("model is ", model, flush=True)

    print(
        "rank",
        dist.get_rank(),
        "model.var_query[0,0,0]",
        model.var_query[0, 0, 0],
        "model.head[0].weight",
        model.head[0].weight[0, 0],
        "pos_embed[0,0,0]",
        model.pos_embed[0, 0, 0],
        "pos_embed[0,0,1]",
        model.pos_embed[0, 0, 1],
        "conv_out.weight",
        model.conv_out.weight[0, 0, 0, 0],
        flush=True,
    )

    # Set random seed for reproducibility
    seed_everything(0)

    # Configure automatic layer wrapping for FSDP
    # This determines which layers should be wrapped for sharding
    if preset == "vit" or preset == "res_slimvit":

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Block,
                Sequential,
            },
        )

        check_fn = lambda submodule: isinstance(submodule, Block) or isinstance(
            submodule, Sequential
        )

    if data_type == "float32":
        precision_dt = torch.float32
    elif data_type == "bfloat16":
        precision_dt = torch.bfloat16
    else:
        raise RuntimeError("Data type not supported")

    # Configure mixed precision policy for memory efficiency
    bfloatPolicy = MixedPrecision(
        param_dtype=precision_dt,
        # Gradient communication precision.
        reduce_dtype=precision_dt,
        # Buffer precision.
        buffer_dtype=precision_dt,
    )

    # Wrap model with Fully Sharded Data Parallel for distributed training
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

    # Apply activation checkpointing to reduce memory usage
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )


    # Update spatial resolution, image size, and number of variables to model
    # based on datasets
    in_shape, _ = data_module.get_data_dims()
            
    print("in_shape is ",in_shape,flush=True)

    _,_, in_height, in_width = in_shape

    with FSDP.summon_full_params(model):
        model.data_config(
            spatial_resolution[data_key],
            (in_height, in_width),
            len(in_vars),
            len(out_vars),
        )



    # Set the model to evaluation mode (disable dropout, batch norm training, etc.)
    model.eval()

    
    # Run visualization on specified sample and variable
    # Note: All ranks must participate in visualization due to potential distributed operations

    with torch.no_grad():
        cl.utils.visualize.visualize_at_index(
            model,
            data_module,
            dm_vis,
            out_list=out_vars,
            in_transform=denorm,
            out_transform=denorm,
            variable=args.variable,  # Variable to visualize
            src=data_key,
            device=device,
            div=div,
            overlap=overlap,
            index=args.index,  # Sample index to visualize
            tensor_par_size=tensor_par_size,
            tensor_par_group=tensor_par_group,
        )

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
