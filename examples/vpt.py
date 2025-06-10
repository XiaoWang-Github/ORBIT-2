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
import random
import time
import numpy as np
import yaml
import torch.nn as nn
import torch.nn.functional as F

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS
)
from climate_learn.models.hub.components.vit_blocks import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock
)
from climate_learn.utils.fused_attn import FusedAttn
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from climate_learn.dist.profile import *

def init_par_groups(data_par_size, tensor_par_size, seq_par_size, fsdp_size, simple_ddp_size, num_heads):
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()

    assert seq_par_size == 1, "Sequence parallelism not implemented"
    assert (data_par_size * seq_par_size * tensor_par_size) == world_size, "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
    assert (num_heads % tensor_par_size) == 0, "model heads % tensor parallel size must be 0"

    tensor_par_group = None
    for i in range(data_par_size * seq_par_size):
        ranks = [j for j in range(i*tensor_par_size, (i+1)*tensor_par_size)]
        if world_rank == 0:
            print("i ", i, " data_par_size ", data_par_size, " SEQ_PAR_SIZE ", seq_par_size, " TENSOR_PAR_SIZE ", tensor_par_size, " tensor_par_group ranks ", ranks)
        group = dist.new_group(ranks)
        if world_rank in ranks:
            tensor_par_group = group

    seq_par_group = None
    for t in range(data_par_size):
        for i in range(tensor_par_size):
            ranks = [t*tensor_par_size*seq_par_size + i + j*tensor_par_size for j in range(seq_par_size)]
            if world_rank == 0:
                print("i ", i, " data_par_size ", data_par_size, " SEQ_PAR_SIZE ", seq_par_size, " TENSOR_PAR_SIZE ", tensor_par_size, " seq_par_group ranks ", ranks, flush=True)
            group = dist.new_group(ranks)
            if world_rank in ranks:
                seq_par_group = group

    data_par_group = None
    fsdp_group = None
    simple_ddp_group = None
    for i in range(tensor_par_size * seq_par_size):
        ranks = [i + j*tensor_par_size * seq_par_size for j in range(data_par_size)]
        for k in range(simple_ddp_size):
            fsdp_begin_idx = k*fsdp_size
            fsdp_end_idx = (k+1)*fsdp_size
            fsdp_ranks = ranks[fsdp_begin_idx:fsdp_end_idx]
            if world_rank == 0:
                print("i ", i, " data_par_size ", data_par_size, " SEQ_PAR_SIZE ", seq_par_size, " TENSOR_PAR_SIZE ", tensor_par_size, " fsdp_ranks", fsdp_ranks)
            group = dist.new_group(fsdp_ranks)
            if world_rank in fsdp_ranks:
                fsdp_group = group

        for k in range(fsdp_size):
            simple_ddp_begin_idx = k
            simple_ddp_end_idx = len(ranks)
            simple_ddp_ranks = ranks[simple_ddp_begin_idx:simple_ddp_end_idx:fsdp_size]
            if world_rank == 0:
                print("i ", i, " data_par_size ", data_par_size, " SEQ_PAR_SIZE ", seq_par_size, " TENSOR_PAR_SIZE ", tensor_par_size, " simple_ddp_ranks", simple_ddp_ranks)
            group = dist.new_group(simple_ddp_ranks)
            if world_rank in simple_ddp_ranks:
                simple_ddp_group = group

        if world_rank == 0:
            print("i ", i, " data_par_size ", data_par_size, " SEQ_PAR_SIZE ", seq_par_size, " TENSOR_PAR_SIZE ", tensor_par_size, " data_par_group ranks ", ranks)
        group = dist.new_group(ranks)
        if world_rank in ranks:
            data_par_group = group

    data_seq_ort_group = None
    for i in range(tensor_par_size):
        ranks = [i + tensor_par_size*j for j in range(data_par_size * seq_par_size)]
        if world_rank == 0:
            print("i ", i, " data_par_size ", data_par_size, " SEQ_PAR_SIZE ", seq_par_size, " TENSOR_PAR_SIZE ", tensor_par_size, " data_seq_ort_group ranks ", ranks)
        group = dist.new_group(ranks)
        if world_rank in ranks:
            data_seq_ort_group = group

    return seq_par_group, data_par_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group

def clip_replace_constant(y, yhat, out_variables):
    prcp_index = out_variables.index("total_precipitation_24hr")
    for i in range(yhat.shape[1]):
        if i==prcp_index:
            torch.clamp_(yhat[:,prcp_index,:,:], min=0.0)

    for i in range(yhat.shape[1]):
        # if constant replace with ground-truth value
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class VisualPromptTuning(nn.Module):
    def __init__(self, model, base_num_prompts=10, prompt_dim=768):
        super().__init__()
        self.model = model
        self.base_num_prompts = base_num_prompts
        self.prompt_dim = prompt_dim
        
        # 冻结预训练模型参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 初始化基础prompt tokens
        self.base_prompt_tokens = nn.Parameter(
            torch.randn(1, base_num_prompts, prompt_dim)
        )
        
        # 添加尺度感知的投影层
        self.scale_projection = nn.Linear(2, prompt_dim)  # 2表示经纬度信息
        
    def forward(self, x, in_variables, out_variables, lat=None, lon=None):
        B = x.shape[0]
        
        # 1. 动态调整prompt数量
        num_patches = x.shape[1]
        # 根据patch数量调整prompt数量，保持一定的比例
        num_prompts = max(self.base_num_prompts, num_patches // 10)
        
        # 使用插值调整prompt数量
        if num_prompts > self.base_num_prompts:
            prompts = F.interpolate(
                self.base_prompt_tokens.transpose(1, 2),
                size=num_prompts,
                mode='linear'
            ).transpose(1, 2)
        else:
            prompts = self.base_prompt_tokens
            
        # 2. 添加物理尺度信息
        if lat is not None and lon is not None:
            # 计算每个patch的物理尺度
            patch_scale = self.calculate_patch_scale(lat, lon)
            # 根据物理尺度调整prompt
            scale_embedding = self.scale_projection(patch_scale)
            # 扩展scale_embedding到与prompts相同的维度
            scale_embedding = scale_embedding.unsqueeze(0).expand(B, -1, -1)
            prompts = prompts + scale_embedding
            
        prompts = prompts.expand(B, -1, -1)
        x = torch.cat([prompts, x], dim=1)
        
        return self.model(x, in_variables, out_variables)
        
    def calculate_patch_scale(self, lat, lon):
        """
        计算每个patch代表的物理尺度
        lat, lon: 输入数据的经纬度信息
        返回: 每个patch的物理尺度特征 [num_patches, 2]
        """
        # 计算每个patch的经纬度范围
        lat_diff = torch.abs(lat[1:] - lat[:-1])
        lon_diff = torch.abs(lon[1:] - lon[:-1])
        
        # 创建网格
        lat_grid, lon_grid = torch.meshgrid(lat_diff, lon_diff)
        
        # 将网格展平为特征
        scale_features = torch.stack([lat_grid.flatten(), lon_grid.flatten()], dim=1)
        
        return scale_features

def load_checkpoint_pretrain(model, checkpoint_path, pretrain_path, cp_save_path, tensor_par_size=1, tensor_par_group=None):
    world_rank = dist.get_rank()
    local_rank = int(os.environ['SLURM_LOCALID'])

    # 加载预训练模型
    if pretrain_path is not None and world_rank < tensor_par_size:
        if tensor_par_size > 1:
            pretrain_path = pretrain_path + "_" + "rank" + "_" + str(world_rank)

        if os.path.exists(pretrain_path):
            print("world_rank", world_rank, "load pretrained model", pretrain_path, " Pretrain path found.", flush=True)
            _load_pretrained_weights(model, pretrain_path, device, world_rank)
            # 将模型包装为 VPT 模型
            model = VisualPromptTuning(model)
        else:
            print("resume from pretrained model was set to True. But the pretrained model path does not exist.", flush=True)
            sys.exit("pretrain path does not exist")

def _load_pretrained_weights(model, pretrain_path, device, world_rank):
    map_location = 'cpu'
    checkpoint = torch.load(pretrain_path, map_location=map_location)
    pretrain_model = checkpoint["model_state_dict"]
    del checkpoint

    state_dict = model.state_dict()
    
    if torch.distributed.get_rank() == 0:
        for k in list(pretrain_model.keys()):
            print("Pretrained model before deletion. Name ", k, "shape", pretrain_model[k].shape, flush=True)

    for k in list(pretrain_model.keys()):
        if k not in state_dict.keys():
            print(f"Removing key {k} from pretrained checkpoint: no exist")
            del pretrain_model[k]
        elif pretrain_model[k].shape != state_dict[k].shape:
            if k == "pos_embed":
                print("interpolate positional embedding", flush=True)
                interpolate_pos_embed(model, pretrain_model, new_size=model.img_size)
            else:
                print(f"Removing key {k} from pretrained checkpoint: no matching shape")
                del pretrain_model[k]

    msg = model.load_state_dict(pretrain_model, strict=False)
    print(msg)
    del pretrain_model

def training_step(batch, batch_idx, net, device, var_weights, train_loss, data_module) -> torch.Tensor:
    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
    
    # 获取经纬度信息并转换为 PyTorch tensor
    lat, lon = data_module.get_lat_lon()
    lat = torch.from_numpy(lat).to(device)  # 将 numpy 数组转换为 tensor
    lon = torch.from_numpy(lon).to(device)  # 将 numpy 数组转换为 tensor
    
    # 使用 VPT 模型进行前向传播，传入经纬度信息
    yhat = net.forward(x, in_variables, out_variables, lat, lon)
    yhat = clip_replace_constant(y, yhat, out_variables)

    if y.size(dim=2) != yhat.size(dim=2) or y.size(dim=3) != yhat.size(dim=3):
        losses = train_loss(yhat, y[:,:,0:yhat.size(dim=2),0:yhat.size(dim=3)], var_names=out_variables, var_weights=var_weights)
    else:
        losses = train_loss(yhat, y, var_names=out_variables, var_weights=var_weights)
    
    loss_name = getattr(train_loss, "name", "loss")
    if losses.dim() == 0:
        loss = losses
    else:
        loss = losses[-1]
        
    return loss

# ... [保持其他函数不变，包括 init_par_groups, clip_replace_constant, validation_step, evaluate_func, seed_everything] ...

def main(device):
    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = dist.get_rank()
    local_rank = int(os.environ['SLURM_LOCALID'])

    print("world_size", world_size, "world_rank", world_rank, "local_rank", local_rank, flush=True)

    config_path = sys.argv[1]
    if world_rank == 0:
        print("config_path", config_path, flush=True)

    conf = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # 加载配置参数
    max_epochs = conf['trainer']['max_epochs']
    checkpoint_path = conf['trainer']['checkpoint']
    batch_size = conf['trainer']['batch_size']
    num_workers = conf['trainer']['num_workers']
    buffer_size = conf['trainer']['buffer_size']
    data_type = conf['trainer']['data_type']
    train_loss = conf['trainer']['train_loss']
    pretrain_path = conf['trainer']['pretrain']

    fsdp_size = conf['parallelism']['fsdp']
    simple_ddp_size = conf['parallelism']['simple_ddp']
    tensor_par_size = conf['parallelism']['tensor_par']
    seq_par_size = conf['parallelism']['seq_par']

    try:
        do_tiling = conf['tiling']['do_tiling']
        if do_tiling:
            div = conf['tiling']['div']
            overlap = conf['tiling']['overlap']
        else:
            div = 1
            overlap = 0
    except:
        do_tiling = False
        div = 1
        overlap = 0

    low_res_dir = conf['data']['low_res_dir']
    high_res_dir = conf['data']['high_res_dir']
    preset = conf['model']['preset']
    var_weights = conf['data']['var_weights']
    dict_out_variables = conf['data']['dict_out_variables']
    dict_in_variables = conf['data']['dict_in_variables']
    default_vars = conf['data']['default_vars']
    spatial_resolution = conf['data']['spatial_resolution']

    lr = float(conf['model']['lr'])
    beta_1 = float(conf['model']['beta_1'])
    beta_2 = float(conf['model']['beta_2'])
    weight_decay = float(conf['model']['weight_decay'])
    warmup_epochs = conf['model']['warmup_epochs']
    warmup_start_lr = float(conf['model']['warmup_start_lr'])
    eta_min = float(conf['model']['eta_min'])

    superres_mag = conf['model']['superres_mag']
    cnn_ratio = conf['model']['cnn_ratio']
    patch_size = conf['model']['patch_size']
    embed_dim = conf['model']['embed_dim']
    depth = conf['model']['depth']
    decoder_depth = conf['model']['decoder_depth']
    num_heads = conf['model']['num_heads']
    mlp_ratio = conf['model']['mlp_ratio']
    drop_path = conf['model']['drop_path']
    drop_rate = conf['model']['drop_rate']

    data_par_size = fsdp_size * simple_ddp_size

    if world_rank == 0:
        print("max_epochs", max_epochs, " ", checkpoint_path, " ", pretrain_path, " ", low_res_dir, " ", high_res_dir, "spatial_resolution", spatial_resolution, "default_vars", default_vars, "preset", preset, "lr", lr, "beta_1", beta_1, "beta_2", beta_2, "weight_decay", weight_decay, "warmup_epochs", warmup_epochs, "warmup_start_lr", warmup_start_lr, "eta_min", eta_min, "superres_mag", superres_mag, "cnn_ratio", cnn_ratio, "patch_size", patch_size, "embed_dim", embed_dim, "depth", depth, "decoder_depth", decoder_depth, "num_heads", num_heads, "mlp_ratio", mlp_ratio, "drop_path", drop_path, "drop_rate", drop_rate, "batch_size", batch_size, "num_workers", num_workers, "buffer_size", buffer_size, "data_type", data_type, "train_loss", train_loss, flush=True)
        print("data_par_size", data_par_size, "fsdp_size", fsdp_size, "simple_ddp_size", simple_ddp_size, "tensor_par_size", tensor_par_size, "seq_par_size", seq_par_size, "division", div, "overlap", overlap, flush=True)

    # 初始化并行组
    seq_par_group, data_par_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(
        data_par_size=data_par_size,
        tensor_par_size=tensor_par_size,
        seq_par_size=seq_par_size,
        fsdp_size=fsdp_size,
        simple_ddp_size=simple_ddp_size,
        num_heads=num_heads
    )

    if data_type == "bfloat16":
        FusedAttn_option = FusedAttn.CK
    else:
        FusedAttn_option = FusedAttn.DEFAULT

    model_kwargs = {
        'default_vars': default_vars,
        'superres_mag': superres_mag,
        'cnn_ratio': cnn_ratio,
        'patch_size': patch_size,
        'embed_dim': embed_dim,
        'depth': depth,
        'decoder_depth': decoder_depth,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'drop_path': drop_path,
        'drop_rate': drop_rate,
        'tensor_par_size': tensor_par_size,
        'tensor_par_group': tensor_par_group,
        "FusedAttn_option": FusedAttn_option
    }

    if world_rank == 0:
        print("model_kwargs", model_kwargs, flush=True)

    if preset != "vit" and preset != "res_slimvit":
        print("Only supports vit or residual slim vit training.", flush=True)
        sys.exit("Not vit or res_slimvit architecture")

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

    while (epoch_start + interval_epochs) < max_epochs:
        for data_key in low_res_dir.keys():
            in_vars = dict_in_variables[data_key]
            out_vars = dict_out_variables[data_key]

            if world_rank == 0:
                print("***************************", flush=True)
                print("data_key is ", data_key, flush=True)
                print("in_vars", in_vars, flush=True)
                print("out_vars", out_vars, flush=True)
                print("default_vars", default_vars, flush=True)
                print("before data_module torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(device)/1024/1024/1024), flush=True)

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
                        print("Overlap must be adjusted to accomodate patch_size of the Transformer. Need to increase the overlap by ", (yinp % patch_size))
                        sys.exit("Please increase the overlap accordingly to the instructions in the print message")

            if world_rank == 0:
                print("after data_module torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(device)/1024/1024/1024), flush=True)

            if first_time_bool:
                model, train_loss, val_losses, test_losses, train_transform, val_transforms, test_transforms = cl.load_downscaling_module(
                    device,
                    model=model,
                    data_module=data_module,
                    architecture=preset,
                    train_loss=train_loss,
                    model_kwargs=model_kwargs
                )

                if dist.get_rank() == 0:
                    print("train_loss", train_loss, "train_transform", train_transform, "val_losses", val_losses, "val_transforms", val_transforms, flush=True)

                model = model.to(device)

                if torch.distributed.get_rank() == 0:
                    print("before load_checkpoint_pretrain model is", flush=True)
                    for name, param in model.named_parameters():
                        print(name, param.data.shape)

                load_checkpoint_pretrain(model, checkpoint_path, pretrain_path, cp_save_path, tensor_par_size=tensor_par_size, tensor_par_group=tensor_par_group)

                if torch.distributed.get_rank() == 0:
                    print("after load_checkpoint_pretrain model is", flush=True)
                    for name, param in model.named_parameters():
                        print(name, param.data.shape)

                seed_everything(0)

                if preset == "vit" or preset == "res_slimvit":
                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls={
                            Block, Sequential
                        },
                    )

                    check_fn = lambda submodule: isinstance(submodule, Block) or isinstance(submodule, Sequential)

                if data_type == "float32":
                    precision_dt = torch.float32
                elif data_type == "bfloat16":
                    precision_dt = torch.bfloat16
                else:
                    raise RuntimeError("Data type not supported")

                bfloatPolicy = MixedPrecision(
                    param_dtype=precision_dt,
                    reduce_dtype=precision_dt,
                    buffer_dtype=precision_dt,
                )

                if fsdp_size > 1 and simple_ddp_size > 1:
                    print("enter hybrid FSDP", flush=True)
                    model = FSDP(model, device_id=local_rank, process_group=(fsdp_group, simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, auto_wrap_policy=auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers=False)
                elif fsdp_size > 1 and simple_ddp_size == 1:
                    print("enter fully sharded FSDP", flush=True)
                    model = FSDP(model, device_id=local_rank, process_group=fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy=auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers=False)
                else:
                    print("enter NO SHARD only,", flush=True)
                    model = FSDP(model, device_id=local_rank, process_group=simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy=auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers=False)

            in_shape, _ = data_module.get_data_dims()
            _, in_height, in_width = in_shape[1:]

            with FSDP.summon_full_params(model):
                model.data_config(spatial_resolution[data_key], (in_height, in_width), len(in_vars), len(out_vars))

            if first_time_bool:
                apply_activation_checkpointing(
                    model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
                )

                if isinstance(model, VisualPromptTuning):
                    optimizer = torch.optim.AdamW(
                        [
                            {'params': model.base_prompt_tokens},
                            {'params': model.scale_projection.parameters()}
                        ],
                        lr=lr * 10,  # 通常 prompt 需要更大的学习率
                        weight_decay=weight_decay
                    )
                else:
                    optimizer = cl.load_optimizer(
                        model, "adamw", {"lr": lr, "weight_decay": weight_decay, "betas": (beta_1, beta_2)}
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
                    print("optimizer resume from checkpoint", checkpoint_path, " Checkpoint path found.", flush=True)
                    src_rank = world_rank - tensor_par_size * dist.get_rank(group=data_seq_ort_group)
                    map_location = 'cpu'
                    if tensor_par_size > 1:
                        checkpoint_path = checkpoint_path + "_" + "rank" + "_" + str(src_rank)

                    checkpoint = torch.load(checkpoint_path, map_location=map_location)
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    epoch_start = checkpoint['epoch'] + 1
                    del checkpoint

            lat, lon = data_module.get_lat_lon()
            train_dataloader = data_module.train_dataloader()
            val_dataloader = data_module.val_dataloader()

            epoch_end = epoch_start + interval_epochs
            epoch_end = epoch_end if epoch_end < max_epochs else max_epochs

            for epoch in range(epoch_start, epoch_end):
                model.train()
                loss = 0.0
                epoch_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if world_rank == 0:
                    print("epoch ", epoch, flush=True)

                for batch_idx, batch in enumerate(train_dataloader):
                    if world_rank == 0:
                        torch.cuda.synchronize(device=device)
                        tic1 = time.perf_counter()

                    loss = training_step(batch, batch_idx, model, device, var_weights, train_loss, data_module)
                    epoch_loss += loss.detach()

                    if world_rank < tensor_par_size:
                        print("epoch: ", epoch, "batch_idx", batch_idx, "world_rank", world_rank, " loss ", loss, flush=True)

                    optimizer.zero_grad()

                    if data_type == "float32":
                        loss.backward()
                        optimizer.step()
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler._scale < min_scale:
                            scaler._scale = torch.tensor(min_scale).to(scaler._scale)

                    if world_rank == 0:
                        print("rank", world_rank, "batch_idx", batch_idx, "get_lr ", scheduler.get_lr(), "after optimizer step torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(device)/1024/1024/1024), flush=True)

                    if world_rank == 0:
                        torch.cuda.synchronize(device=device)
                        tic4 = time.perf_counter()
                        print(f"my rank {dist.get_rank()}. tic4-tic1 in {(tic4-tic1):0.4f} seconds\n", flush=True)

                scheduler.step()

                if world_rank == 0:
                    print("epoch: ", epoch, " epoch_loss ", epoch_loss, flush=True)

                if world_rank == 0:
                    isExist = os.path.exists(cp_save_path)
                    if not isExist:
                        os.makedirs(cp_save_path)
                        print("The new checkpoint save directory is created!")

                if world_rank == 0:
                    print("rank", world_rank, "Before torch.save torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(device)/1024/1024/1024), flush=True)

                model_states = model.state_dict()
                optimizer_states = optimizer.state_dict()
                scheduler_states = scheduler.state_dict()

                if world_rank < tensor_par_size:
                    if tensor_par_size > 1:
                        file_name = cp_save_path + "/" + "interm" + "_epoch_" + str(epoch) + ".ckpt" + "_" + "rank" + "_" + str(world_rank)
                    else:
                        file_name = cp_save_path + "/" + "interm" + "_epoch_" + str(epoch) + ".ckpt"

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_states,
                        'optimizer_state_dict': optimizer_states,
                        'scheduler_state_dict': scheduler_states,
                    }, file_name)

                print("rank", world_rank, "After torch.save torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(device)/1024/1024/1024), flush=True)

                dist.barrier(device_ids=[local_rank])
                del model_states
                del optimizer_states
                del scheduler_states

            epoch_start = epoch_end

            if first_time_bool:
                first_time_bool = False

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

    print("Using dist.init_process_group. world_size ", world_size, flush=True)

    main(device)

    dist.destroy_process_group()