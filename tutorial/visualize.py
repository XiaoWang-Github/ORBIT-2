import climate_learn as cl
import torch
import os
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
from datetime import timedelta
import sys
import random
import time
import numpy as np
import yaml

from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS
)
from climate_learn.models.hub.components.vit_blocks import Block
from torch.nn import Sequential
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from climate_learn.utils.fused_attn import FusedAttn




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_checkpoint_pretrain(model, pretrain_path, tensor_par_size=1,tensor_par_group=None):
    world_rank = dist.get_rank()
    local_rank = int(os.environ['SLURM_LOCALID'])

    if tensor_par_size >1:
        pretrain_path = pretrain_path+"_"+"rank"+"_"+str(world_rank)



    print("world_rank",world_rank,"pretrain_path",pretrain_path,flush=True)

    #load pretrained model
    if world_rank < tensor_par_size:
        if os.path.exists(pretrain_path):
            print("world_rank",world_rank,"load pretrained model",pretrain_path," Pretrain path found.",flush=True)
            _load_pretrained_weights(model,pretrain_path,device,world_rank)  
        else:
            print("resume from pretrained model was set to True. But the pretrained model path does not exist.",flush=True)
            sys.exit("pretrain path does not exist")

    dist.barrier(device_ids= [local_rank])

 

def _load_pretrained_weights(model, pretrain_path, device,world_rank):
    # map_location = 'cuda:'+str(device)
    map_location = 'cpu'
    checkpoint = torch.load(pretrain_path, map_location=map_location)

    print("Loading pre-trained checkpoint from: %s" % pretrain_path)
    pretrain_model = checkpoint["model_state_dict"]

    del checkpoint


    state_dict = model.state_dict()
  
    if torch.distributed.get_rank()==0: 
        for k in list(pretrain_model.keys()):
            print("Pretrained model before deletion. Name ",k,"shape",pretrain_model[k].shape,flush=True)


    # checkpoint_keys = list(pretrain_model.keys())
    for k in list(pretrain_model.keys()):  #in pre-train model weights, but not fine-tuning model
        if k not in state_dict.keys():
            print(f"Removing key {k} from pretrained checkpoint: no exist")
            del pretrain_model[k]
        elif pretrain_model[k].shape != state_dict[k].shape:  #if pre-train and fine-tune model weights dimension doesn't match
            if k =="pos_embed":
                print("interpolate positional embedding",flush=True)
                interpolate_pos_embed(model, pretrain_model, new_size=model.img_size)
            else:
                print(f"Removing key {k} from pretrained checkpoint: no matching shape", pretrain_model[k].shape, state_dict[k].shape)
                del pretrain_model[k]
  
#    for k in list( checkpoint_model.keys()):
#        print("after deletion. Name ",k,flush=True)

    # load pre-trained model
    msg = model.load_state_dict(pretrain_model, strict=False)
    print(msg)
    del pretrain_model



"""
Setup sequence, data, tensor model, and sequence_plus_data parallel groups
"""
def init_par_groups(data_par_size, tensor_par_size, seq_par_size, fsdp_size, simple_ddp_size, num_heads):

    world_size = torch.distributed.get_world_size()

    assert seq_par_size ==1, "Sequence parallelism not implemented"

    assert (data_par_size * seq_par_size * tensor_par_size)==world_size, "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
    assert (num_heads % tensor_par_size) ==0, "model heads % tensor parallel size must be 0"



    tensor_par_group = None

    for i in range(data_par_size *seq_par_size):
        ranks = [j for j in range(i*tensor_par_size,(i+1)*tensor_par_size)]

        if world_rank==0:
            print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," tensor_par_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:
            tensor_par_group = group




    seq_par_group = None

    for t in range(data_par_size):
        for i in range(tensor_par_size):
            ranks = [t*tensor_par_size*seq_par_size+i+j*tensor_par_size for j in range(seq_par_size)]

            if world_rank==0:
                print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size, " TENSOR_PAR_SIZE ",tensor_par_size," seq_par_group ranks ",ranks,flush=True)

            group = dist.new_group(ranks)

            if world_rank in ranks:

                seq_par_group = group




    data_par_group = None

    fsdp_group = None

    simple_ddp_group = None

    for i in range(tensor_par_size *seq_par_size):
        ranks = [i+j*tensor_par_size *seq_par_size for j in range(data_par_size)]

        for k in range(simple_ddp_size):
            fsdp_begin_idx = k*fsdp_size
            fsdp_end_idx = (k+1)*fsdp_size
            fsdp_ranks = ranks[fsdp_begin_idx:fsdp_end_idx]

 
            if world_rank==0:
                print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," fsdp_ranks",fsdp_ranks)


            group = dist.new_group(fsdp_ranks)
            if world_rank in fsdp_ranks:
                fsdp_group = group


        for k in range(fsdp_size):
            simple_ddp_begin_idx = k
            simple_ddp_end_idx = len(ranks)
            simple_ddp_ranks = ranks[simple_ddp_begin_idx:simple_ddp_end_idx:fsdp_size]

 
            if world_rank==0:
                print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," simple_ddp_ranks",simple_ddp_ranks)

            group = dist.new_group(simple_ddp_ranks)
            if world_rank in simple_ddp_ranks:
                simple_ddp_group = group
 
        if world_rank==0:
            print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," data_par_group ranks ",ranks)
        group = dist.new_group(ranks)
        if world_rank in ranks:
            data_par_group = group


    data_seq_ort_group = None

    for i in range(tensor_par_size):
        ranks = [i+tensor_par_size*j for j in range(data_par_size * seq_par_size)]

        if world_rank==0:
            print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," data_seq_ort_group ranks ",ranks)
        group = dist.new_group(ranks)

        if world_rank in ranks:
            data_seq_ort_group = group

    return seq_par_group, data_par_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group




os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
os.environ['MASTER_PORT'] = "29500"
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
os.environ['RANK'] = os.environ['SLURM_PROCID']

world_size = int(os.environ['SLURM_NTASKS'])
world_rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])


torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

# assuming we are downscaling geopotential from ERA5

config_path = sys.argv[1]

if world_rank==0:
    print("config_path",config_path,flush=True)

conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

max_epochs=conf['trainer']['max_epochs']
checkpoint_path = conf['trainer']['checkpoint']
batch_size = conf['trainer']['batch_size']
num_workers = conf['trainer']['num_workers']
buffer_size = conf['trainer']['buffer_size']
pretrain_path = conf['trainer']['pretrain']
data_type = "float32"

try:
    do_tiling = conf['tiling']['do_tiling']
    if do_tiling:
        div = conf['tiling']['div']
        overlap = conf['tiling']['overlap']
    else:
        div = 1
        overlap = 0
except:
    print("Tiling parameter not found. Using default: no tiling", flush=True)
    do_tiling = False
    div = 1
    overlap = 0

tensor_par_size = conf['parallelism']['tensor_par']
fsdp_size = world_size //tensor_par_size
simple_ddp_size = 1
seq_par_size = 1

FusedAttn_option = FusedAttn.DEFAULT 

low_res_dir = conf['data']['low_res_dir']
high_res_dir = conf['data']['high_res_dir']
preset = conf['model']['preset']
dict_out_variables = conf['data']['dict_out_variables']
dict_in_variables = conf['data']['dict_in_variables']
default_vars =  conf['data']['default_vars']


lr = float(conf['model']['lr'])
beta_1 = float(conf['model']['beta_1'])
beta_2 = float(conf['model']['beta_2'])
weight_decay = float(conf['model']['weight_decay'])
warmup_epochs =  conf['model']['warmup_epochs']
warmup_start_lr =  float(conf['model']['warmup_start_lr'])
eta_min =  float(conf['model']['eta_min'])

superres_mag = conf['model']['superres_mag']
cnn_ratio = conf['model']['cnn_ratio']
patch_size =  conf['model']['patch_size']
embed_dim = conf['model']['embed_dim']
depth = conf['model']['depth']
decoder_depth = conf['model']['decoder_depth']
num_heads = conf['model']['num_heads']
mlp_ratio = conf['model']['mlp_ratio']
drop_path = conf['model']['drop_path']
drop_rate = conf['model']['drop_rate']

data_par_size = fsdp_size * simple_ddp_size

if world_rank==0:
    print("max_epochs",max_epochs," ",checkpoint_path," ",pretrain_path," ",low_res_dir," ",high_res_dir,"preset",preset,"dict_out_variables",dict_out_variables,"lr",lr,"beta_1",beta_1,"beta_2",beta_2,"weight_decay",weight_decay,"warmup_epochs",warmup_epochs,"warmup_start_lr",warmup_start_lr,"eta_min",eta_min,"superres_mag",superres_mag,"cnn_ratio",cnn_ratio,"patch_size",patch_size,"embed_dim",embed_dim,"depth",depth,"decoder_depth",decoder_depth,"num_heads",num_heads,"mlp_ratio",mlp_ratio,"drop_path",drop_path,"drop_rate",drop_rate,"batch_size",batch_size,"num_workers",num_workers,"buffer_size",buffer_size,flush=True)
    print("data_par_size",data_par_size,"fsdp_size",fsdp_size,"simple_ddp_size",simple_ddp_size,"tensor_par_size",tensor_par_size,"seq_par_size",seq_par_size,"FusedAttn_option",FusedAttn_option, flush=True)


#initialize parallelism groups
_, data_par_group, tensor_par_group, _, fsdp_group, _ = init_par_groups(data_par_size = data_par_size, tensor_par_size = tensor_par_size, seq_par_size = seq_par_size, fsdp_size = fsdp_size, simple_ddp_size = simple_ddp_size, num_heads= num_heads)



model_kwargs = {'default_vars':default_vars,'superres_mag':superres_mag,'cnn_ratio':cnn_ratio,'patch_size':patch_size,'embed_dim':embed_dim,'depth':depth,'decoder_depth':decoder_depth,'num_heads':num_heads,'mlp_ratio':mlp_ratio,'drop_path':drop_path,'drop_rate':drop_rate, 'tensor_par_size':tensor_par_size, 'tensor_par_group':tensor_par_group,'FusedAttn_option':FusedAttn_option}


if world_rank==0:
    print("model_kwargs",model_kwargs,flush=True)


if preset!="vit" and preset!="res_slimvit":
    print("Only supports vit or residual slim vit training.",flush=True)
    sys.exit("Not vit or res_slimvit architecture")



# Set up data
data_key = "DAYMET_3"

in_vars = dict_in_variables[data_key]
out_vars = dict_out_variables[data_key]
 

if world_rank==0:
    print("in_vars",in_vars,flush=True)
    print("out_vars",out_vars,flush=True)
 



#load data module
data_module = cl.data.IterDataModule(
    "downscaling",
    low_res_dir[data_key], 
    high_res_dir[data_key],
    in_vars,
    out_vars=out_vars,
    data_par_size = data_par_size,
    data_par_group = data_par_group,
    subsample=1,
    batch_size=1,
    buffer_size=buffer_size,
    num_workers=num_workers,
    div=div,
    overlap=overlap,
).to(device)

data_module.setup()


dm_vis = cl.data.IterDataModule(
    "downscaling",
    low_res_dir[data_key],
    high_res_dir[data_key],
    in_vars,
    out_vars=out_vars,
    data_par_size = data_par_size,
    data_par_group = data_par_group,
    subsample=1,
    batch_size=1,
    buffer_size=buffer_size,
    num_workers=num_workers,
    div=1,
    overlap=0,
).to(device)

dm_vis.setup()


# Set up deep learning model
model, train_loss,val_losses,test_losses,train_transform,val_transforms,test_transforms = cl.load_downscaling_module(device,data_module=data_module, architecture=preset,model_kwargs=model_kwargs)
  
if dist.get_rank()==0:
    print("train_loss",train_loss,"train_transform",train_transform,"img_size",model.img_size,flush=True)
 
model = model.to(device)

#denorm = model.test_target_transforms[0]


denorm = test_transforms[0]


print("denorm is ",denorm,flush=True)

# define your checkpoint path here otherwise it would use the one provided through "pretrain" in yaml
# pretrain_path = "./checkpoints/climate/interm_epoch_6.ckpt"
# pretrain_path = "./checkpoints/climate/finetuning_1b_daymet3_step2/interm_epoch_88.ckpt"

# load from pretrained model weights
load_checkpoint_pretrain(model, pretrain_path,tensor_par_size=tensor_par_size,tensor_par_group=tensor_par_group)
    

if torch.distributed.get_rank()==0:
    print("model is ",model,flush=True)


print("rank",dist.get_rank(),"model.var_query[0,0,0]",model.var_query[0,0,0],"model.head[0].weight",model.head[0].weight[0,0],"pos_embed[0,0,0]",model.pos_embed[0,0,0],"pos_embed[0,0,1]",model.pos_embed[0,0,1],"conv_out.weight",model.conv_out.weight[0,0,0,0],flush=True)

seed_everything(0)



#set up layer wrapping
if preset =="vit" or preset=="res_slimvit":
       
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block, Sequential # < ---- Your Transformer layer class
        },
    )
    
    check_fn = lambda submodule: isinstance(submodule, Block)  or isinstance(submodule,Sequential)
   

if data_type == "float32":
    precision_dt = torch.float32
elif data_type == "bfloat16":
    precision_dt = torch.bfloat16
else:
    raise RuntimeError("Data type not supported") 

#floating point policy
bfloatPolicy = MixedPrecision(
    param_dtype=precision_dt,
    # Gradient communication precision.
    reduce_dtype=precision_dt,
    # Buffer precision.
    buffer_dtype=precision_dt,
)

#fully sharded FSDP
print("enter fully sharded FSDP",flush=True)
model = FSDP(model, device_id = local_rank, process_group= fsdp_group,sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD,auto_wrap_policy = auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False)
    
    
#activation checkpointing
apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
)








# Set the model to evaluation mode
model.eval()

cl.utils.visualize.visualize_at_index(
    model,
    data_module,
    dm_vis,
    out_list=out_vars,
    in_transform=denorm,
    out_transform=denorm,
    variable="2m_temperature_min",
    src=data_key,
    device = device,
    div=div,
    overlap=overlap,
    index=0,  # visualize the first sample of the test set
    tensor_par_size=tensor_par_size,
    tensor_par_group=tensor_par_group,
)

dist.destroy_process_group()
