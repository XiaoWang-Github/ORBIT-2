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

from timm.models.vision_transformer import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock
)

from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed


def load_pretrained_weights(model, pretrained_path, device):
    # map_location = 'cuda:'+str(device)
    map_location = 'cpu'
    checkpoint = torch.load(pretrained_path, map_location=map_location)

    print("Loading pre-trained checkpoint from: %s" % pretrained_path)
    pretrain_model = checkpoint["model_state_dict"]

    del checkpoint


    state_dict = model.state_dict()
   
    for k in list(pretrain_model.keys()):
        print("Pretrained model before deletion. Name ",k,flush=True)


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
data_type = conf['trainer']['data_type']

pretrain_path = conf['trainer']['pretrain']
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
adaptive_patching = conf['model']['adaptive_patching']
if adaptive_patching:
    fixed_length = conf['model']['fixed_length']
    data_key = list(fixed_length.keys())[0]

    physics = conf['model']['physics']
    if physics:
        edge_percentage = conf['model']['edge_percentage']
        grad_deg = conf['model']['grad_deg']
    else:
        edge_percentage = None
        grad_deg = None
    smooth = conf['model']['smooth']
    canny = conf['model']['canny']
    canny_add = conf['model']['canny_add']
else:
    fixed_length = None
    smooth = None
    canny = None
    canny_add = None
    edge_percentage = None
    grad_deg = None


if world_rank==0:
    print("max_epochs",max_epochs," ",checkpoint_path," ",pretrain_path," ",low_res_dir," ",high_res_dir,"preset",preset,"dict_out_variables",dict_out_variables,"lr",lr,"beta_1",beta_1,"beta_2",beta_2,"weight_decay",weight_decay,"warmup_epochs",warmup_epochs,"warmup_start_lr",warmup_start_lr,"eta_min",eta_min,"superres_mag",superres_mag,"cnn_ratio",cnn_ratio,"patch_size",patch_size,"embed_dim",embed_dim,"depth",depth,"decoder_depth",decoder_depth,"num_heads",num_heads,"mlp_ratio",mlp_ratio,"drop_path",drop_path,"drop_rate",drop_rate,"batch_size",batch_size,"num_workers",num_workers,"buffer_size",buffer_size,"adaptive_patching",adaptive_patching,"fixed_length",fixed_length,"smooth",smooth,"canny",canny,"canny_add",canny_add,"physics",physics,"edge_percentage",edge_percentage,'grad_deg',grad_deg,flush=True)


if adaptive_patching:
    model_kwargs = {'default_vars':default_vars,'superres_mag':superres_mag,'cnn_ratio':cnn_ratio,'patch_size':patch_size,'embed_dim':embed_dim,'depth':depth,'decoder_depth':decoder_depth,'num_heads':num_heads,'mlp_ratio':mlp_ratio,'drop_path':drop_path,'drop_rate':drop_rate, 'adaptive_patching':adaptive_patching,'fixed_length':fixed_length[data_key],'smooth':smooth[data_key],'canny':canny[data_key],'canny_add':canny_add[data_key],'physics':physics,'edge_percentage':edge_percentage,'grad_deg':grad_deg}
else:
    model_kwargs = {'default_vars':default_vars,'superres_mag':superres_mag,'cnn_ratio':cnn_ratio,'patch_size':patch_size,'embed_dim':embed_dim,'depth':depth,'decoder_depth':decoder_depth,'num_heads':num_heads,'mlp_ratio':mlp_ratio,'drop_path':drop_path,'drop_rate':drop_rate, 'adaptive_patching':adaptive_patching,'fixed_length':fixed_length,'smooth':smooth,'canny':canny,'canny_add':canny_add,'physics':physics,'edge_percentage':edge_percentage,'grad_deg':grad_deg}


if world_rank==0:
    print("model_kwargs",model_kwargs,flush=True)


if preset!="vit" and preset!="res_slimvit":
    print("Only supports vit or residual slim vit training.",flush=True)
    sys.exit("Not vit or res_slimvit architecture")



# Set up data
#data_key = "ERA5_1"
data_key = "PRISM"

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
    subsample=1,
    batch_size=1,
    buffer_size=buffer_size,
    num_workers=num_workers,
).to(device)

data_module.setup()


# Set up deep learning model
model, train_loss,val_losses,test_losses,train_transform,val_transforms,test_transforms = cl.load_downscaling_module(device,data_module=data_module, architecture=preset,model_kwargs=model_kwargs)
  
if dist.get_rank()==0:
    print("train_loss",train_loss,"train_transform",train_transform,"img_size",model.img_size,flush=True)
 

#denorm = model.test_target_transforms[0]


denorm = test_transforms[0]


print("denorm is ",denorm,flush=True)

checkpoint_file = checkpoint_path


#load pretrained model
if os.path.exists(checkpoint_file):
    print("load pretrained model",checkpoint_file," Pretrain path found.",flush=True)
    load_pretrained_weights(model,checkpoint_file,device)  
else:
    print("resume from pretrained model was set to True. But the pretrained model path does not exist.",flush=True)
    sys.exit("pretrain path does not exist")




model = model.to(device)



if torch.distributed.get_rank()==0:
    print("model is ",model,flush=True)




# Set the model to evaluation mode
model.eval()

cl.utils.visualize.visualize_at_index(
    model,
    data_module,
    out_list=out_vars,
    in_transform=denorm,
    out_transform=denorm,
    #variable="2m_temperature_max",
    #variable="2m_temperature_min",
    variable="total_precipitation_24hr",
    src=data_key,
    device = device,
    index=0,  # visualize the first sample of the test set
)

dist.destroy_process_group()
