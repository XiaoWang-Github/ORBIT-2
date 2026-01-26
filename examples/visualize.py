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
import torch.distributed as dist
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
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from torch.nn import Sequential
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from climate_learn.utils.fused_attn import FusedAttn
from climate_learn.utils import quantization_utils
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


def _load_pretrained_weights(model, pretrain_path, device, world_rank):
    if world_rank == 0:
        print(
            "world_rank",
            world_rank,
            "load pretrained model",
            pretrain_path,
            " Pretrain path found.",
            flush=True,
        )
    
    # Load checkpoint
    checkpoint = torch.load(pretrain_path, map_location="cpu")
    
    # Handle both full checkpoint and state_dict only
    if world_rank == 0:
        print(f"Checkpoint keys: {list(checkpoint.keys())}", flush=True)

    if "model_state_dict" in checkpoint:
        pretrain_model = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        pretrain_model = checkpoint["state_dict"]
    else:
        pretrain_model = checkpoint

    # Clean up state dict keys if needed (remove 'module.' prefix)
    new_state_dict = {}
    for k, v in pretrain_model.items():
        name = k.replace("module.", "") if "module." in k else k
        # Handle FSDP prefix removal if present in checkpoint
        name = name.replace("_fsdp_wrapped_module.", "") if "_fsdp_wrapped_module." in name else name
        new_state_dict[name] = v
    pretrain_model = new_state_dict

    state_dict = model.state_dict()
    
    # Handle shape mismatches and interpolation
    for k in list(pretrain_model.keys()):
        if k not in state_dict.keys():
            if world_rank == 0:
                # print(f"Removing key {k} from pretrained checkpoint: no exist in model", flush=True)
                pass
            del pretrain_model[k]
        elif pretrain_model[k].shape != state_dict[k].shape:
            if k == "pos_embed":
                if world_rank == 0:
                    print("interpolate positional embedding", flush=True)
                interpolate_pos_embed(model, pretrain_model, new_size=model.img_size)
            else:
                if world_rank == 0:
                    print(
                        f"Removing key {k} from pretrained checkpoint: no matching shape {pretrain_model[k].shape} vs {state_dict[k].shape}",
                        flush=True,
                    )
                del pretrain_model[k]

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(pretrain_model, strict=False)
    
    if world_rank == 0:
        print(f"Missing keys: {len(missing_keys)}", flush=True)


def load_pretrained_weights(
    model, pretrain_path, device, tensor_par_size=1, tensor_par_group=None
):
    """Load pretrained model weights for visualization.

    Args:
        model: PyTorch model to load weights into
        pretrain_path (str): Path to the pretrained model checkpoint
        device: Device to load the model on
        tensor_par_size (int): Size of tensor parallelism (default: 1)
        tensor_par_group: Process group for tensor parallelism (default: None)
        auto_wrap_policy: FSDP auto wrap policy
        
    Returns:
        model: Model (possibly converted to INT8 if QAT checkpoint)
    """
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    # Adjust path for tensor parallel checkpoints
    if tensor_par_size > 1 and pretrain_path is not None:
        pretrain_path = pretrain_path + "_" + "rank" + "_" + str(world_rank)

    # load pretrained model
    if pretrain_path is None:
        print(
            "world_rank",
            world_rank,
            "No pretrained model path provided in config.",
            flush=True,
        )
        sys.exit("pretrain_path is None - please specify pretrain path in config file")
    elif os.path.exists(pretrain_path):
        _load_pretrained_weights(model, pretrain_path, device, world_rank)
        
        # Check if this is a QAT checkpoint that should be converted to INT8
        # Load checkpoint to check metadata (only rank 0 prints, but all ranks convert)
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        quantization_info = checkpoint.get('quantization', {})
        
        if quantization_info.get('enabled'):
            if world_rank == 0:
                print("\n" + "="*80, flush=True)
                print("QAT CHECKPOINT DETECTED", flush=True)
                print("="*80, flush=True)
                print(f"Precision: {quantization_info.get('precision', 'int8')}", flush=True)
                print(f"Method: {quantization_info.get('method', 'qat')}", flush=True)
                print("Converting model to INT8 for inference...", flush=True)
            
            try:
                from climate_learn.utils import qat_utils
                # Set model to eval mode before conversion
                model.eval()
                # Convert QAT model to true INT8 (ALL RANKS)
                model = qat_utils.convert_qat_to_quantized(model)
                # CRITICAL: Force float32 for all parameters after INT8 conversion
                # INT8 ops will still use quantized kernels, but intermediate layers stay float32
                model = model.to(torch.float32)
                if world_rank == 0:
                    print("✓ Successfully converted to INT8 quantized model", flush=True)
                    print("✓ Forced all parameters to float32", flush=True)
                    print("="*80 + "\n", flush=True)
            except Exception as e:
                if world_rank == 0:
                    print(f"WARNING: Failed to convert to INT8: {e}", flush=True)
                    print("Continuing with FP32 model...", flush=True)
                    print("="*80 + "\n", flush=True)
        
        del checkpoint
        
    else:
        print(
            "resume from pretrained model was set to True. But the pretrained model path does not exist.",
            flush=True,
        )
        sys.exit("pretrain path does not exist")

    dist.barrier(device_ids=[local_rank])
    
    return model


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
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization to attention layers (PTQ)",
    )
    parser.add_argument(
        "--quantize-all",
        action="store_true",
        help="Apply INT8 quantization to all layers (not recommended, use --quantize for hybrid strategy)",
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
    
    # Priority: 1) Command line argument, 2) Config pretrain path
    if args.checkpoint:
        pretrain_path = args.checkpoint
        if world_rank == 0:
            print(f"Using checkpoint from command line: {pretrain_path}", flush=True)
    else:
        pretrain_path = conf["trainer"]["pretrain"]
        if world_rank == 0:
            print(f"Using checkpoint from config: {pretrain_path}", flush=True)
    # Get data type from config or command line
    # Command line takes precedence, otherwise use config value
    if args.data_type:
        data_type = args.data_type
        if world_rank == 0:
            print(f"Using data_type from command line: {data_type}", flush=True)
    else:
        data_type = conf["trainer"].get("data_type", "float32")
        if world_rank == 0:
            print(f"Using data_type from config: {data_type}", flush=True)
    
    # Validate data type (only bfloat16 or float32 allowed)
    validate_data_type(data_type)
    
    # Note: For visualization, we may need to force float32 if using bfloat16
    # because some numpy operations don't support bfloat16
    if data_type == "bfloat16":
        if world_rank == 0:
            print("Note: Using bfloat16 for model inference (may force float32 for numpy conversion if needed)", flush=True)

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

    # Set the model to evaluation mode
    model.eval()

    # Load pretrained model weights from checkpoint
    model = load_pretrained_weights(
        model,
        pretrain_path,
        device,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
    )
    
    # Apply precision based on data_type setting AFTER loading checkpoint
    # This ensures all parameters (weights and biases) have consistent dtype
    if data_type == "bfloat16":
        model = model.to(torch.bfloat16)
        if world_rank == 0:
            print("✓ Model converted to bfloat16", flush=True)
    elif data_type == "float32":
        model = model.to(torch.float32)
        if world_rank == 0:
            print("✓ Model using float32", flush=True)

    if torch.distributed.get_rank() == 0:
        print("model is ", model, flush=True)

    # =========================================================================
    # POST-TRAINING QUANTIZATION (PTQ) - Added for hybrid quantization
    # =========================================================================
    if args.quantize or args.quantize_all:
        if world_rank == 0:
            print("\n" + "="*80, flush=True)
            print("POST-TRAINING QUANTIZATION (PTQ) ENABLED", flush=True)
            print("="*80, flush=True)
            
            # Check ROCm environment
            env_info = quantization_utils.check_rocm_quantization_support()
            print("\nEnvironment Information:", flush=True)
            print(f"  PyTorch version: {env_info['pytorch_version']}", flush=True)
            print(f"  CUDA available: {env_info['cuda_available']}", flush=True)
            print(f"  ROCm available: {env_info['rocm_available']}", flush=True)
            if env_info['rocm_available']:
                print(f"  ROCm version: {env_info['rocm_version']}", flush=True)
            print(f"  Device: {env_info['device_name']}", flush=True)
            print(f"  Quantization available: {env_info['quantization_available']}", flush=True)
            
            if not env_info['quantization_available']:
                print("\nWARNING: PyTorch quantization not available!", flush=True)
                print("Skipping quantization...\n", flush=True)
        else:
            # Other ranks need to know if quantization is available
            env_info = quantization_utils.check_rocm_quantization_support()
        
        # Wait for rank 0 to finish printing
        dist.barrier()
        
        if env_info['quantization_available']:
            # Apply quantization
            attention_only = not args.quantize_all
            
            if world_rank == 0:
                if attention_only:
                    print("Applying HYBRID quantization (Attention INT8, CNN FP16/32)...", flush=True)
                else:
                    print("Applying FULL model quantization (all layers INT8)...", flush=True)
            
            # Get tensor_par_size from config for quantization compatibility
            tensor_par_size = conf["parallelism"]["tensor_par"]
            
            # Apply dynamic quantization (will move to CPU, quantize, then try to move back)
            model = quantization_utils.apply_dynamic_quantization(
                model,
                attention_only=attention_only,
                dtype=torch.qint8,
                device=device,
                tensor_par_size=tensor_par_size,
            )
            
            # Print quantization summary
            if world_rank == 0:
                quantization_utils.print_model_quantization_summary(model)
            
            dist.barrier()
    # =========================================================================

    # print(
    #     "rank",
    #     dist.get_rank(),
    #     "model.var_query[0,0,0]",
    #     model.var_query[0, 0, 0],
    #     "model.head[0].weight",
    #     model.head[0].weight()[0, 0] if callable(model.head[0].weight) else model.head[0].weight[0, 0],
    #     "pos_embed[0,0,0]",
    #     model.pos_embed[0, 0, 0],
    #     "pos_embed[0,0,1]",
    #     model.pos_embed[0, 0, 1],
    #     "conv_out.weight",
    #     model.conv_out.weight[0, 0, 0, 0],
    #     flush=True,
    # )

    # Set random seed for reproducibility
    seed_everything(0)


    # Run visualization on specified sample and variable
    # Note: All ranks must participate in visualization due to potential distributed operations
    # Run visualization on all test samples
    # Note: All ranks must participate in visualization due to potential distributed operations
    if world_rank == 0:
        print(f"Starting inference on all test samples...", flush=True)
    
    psnr_list = []
    ssim_list = []
    
    # Iterate over dataloader directly
    for batch_idx, batch in enumerate(data_module.test_dataloader()):
        if world_rank == 0:
            print(f"\nProcessing batch {batch_idx}...", flush=True)
            
        # visualize_batch returns a list of metrics for the batch
        batch_metrics = cl.utils.visualize.visualize_batch(
            model,
            batch,
            data_module,
            out_list=out_vars,
            in_transform=denorm,
            out_transform=denorm,
            variable=args.variable,
            src=data_key,
            device=device,
            div=div,
            overlap=overlap,
            batch_idx=batch_idx,
        )
        
        if world_rank == 0 and batch_metrics:
            for m in batch_metrics:
                if 'psnr' in m:
                    psnr_list.append(m['psnr'])
                if 'ssim' in m:
                    ssim_list.append(m['ssim'])

    if world_rank == 0:
        print("\n" + "="*80, flush=True)
        print("FINAL EVALUATION RESULTS", flush=True)
        print("="*80, flush=True)
        if psnr_list:
            avg_psnr = sum(psnr_list) / len(psnr_list)
            print(f"Average PSNR: {avg_psnr:.6f} (over {len(psnr_list)} samples)", flush=True)
        if ssim_list:
            avg_ssim = sum(ssim_list) / len(ssim_list)
            print(f"Average SSIM: {avg_ssim:.6f} (over {len(ssim_list)} samples)", flush=True)
        print("="*80 + "\n", flush=True)

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
