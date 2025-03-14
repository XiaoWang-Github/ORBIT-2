import yaml
import climate_learn as cl
import torch
import os
# Standard library
from argparse import ArgumentParser

from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    SR_PRESSURE_LEVELS,
)

#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import (
#    EarlyStopping,
#    ModelCheckpoint,
#    RichModelSummary,
#)
#from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
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

import os
import sys
import torch
from pytorch_lightning.strategies import FSDPStrategy
from timm.models.vision_transformer import Block
from pytorch_lightning.callbacks import DeviceStatsMonitor
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from datetime import timedelta


os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
os.environ['MASTER_PORT'] = "29500"
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
os.environ['RANK'] = os.environ['SLURM_PROCID']

world_size = int(os.environ['SLURM_NTASKS'])
world_rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])


# Parse args
parser = ArgumentParser()
parser.add_argument("--lowresdir")
parser.add_argument("--highresdir")
parser.add_argument("--checkpoint_path", type=str, default="./")
parser.add_argument("--outputdir", type=str, default="./")
parser.add_argument("--variable", type=str, default="prcp")
parser.add_argument("--preset", choices=["resnet", "unet", "vit","res_slimvit"], default='res_slimvit', required=False)
parser.add_argument("--landcover", action='store_true')
parser.add_argument("--era5_total_precip", action='store_true')
parser.add_argument("--soil_moist", action='store_true')
parser.add_argument("--expname", type=int)
parser.add_argument("--data_key", type=str, default='DAYMET', choices=['DAYMET', 'ERA5_1', 'ERA5_2', 'PRISM'])
# temporally setting
parser.add_argument("--config_path", type=str)
#parser.add_argument("--epoch", default=None)
args = parser.parse_args()

torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

num_nodes = 1
# assuming we are downscaling geopotential from ERA5
# Read config YAML
config_path = args.config_path
if world_rank==0:
    print("config_path",config_path,flush=True)
conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

# Load config to local variables
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

default_vars =  conf['data']['default_vars']
dict_in_variables = conf['data']['dict_in_variables']
dict_out_variables = conf['data']['dict_out_variables']
spatial_resolution = conf['data']['spatial_resolution']

model_kwargs = {'default_vars':default_vars,'superres_mag':superres_mag,'cnn_ratio':cnn_ratio,'patch_size':patch_size,'embed_dim':embed_dim,'depth':depth,'decoder_depth':decoder_depth,'num_heads':num_heads,'mlp_ratio':mlp_ratio,'drop_path':drop_path,'drop_rate':drop_rate}
print("model_kwargs",model_kwargs,flush=True)

# Setup
data_key = args.data_key 
in_vars = dict_in_variables[data_key]
out_vars = dict_out_variables[data_key]

if args.landcover:
    in_vars.append("landcover")
    print("landcover is added!")

if args.era5_total_precip:
    #in_vars.append('total_precipitation')
    #print("EAR5 total_precipitation is added")
    print("EAR5 total_precipitation is default variable")
    
if args.soil_moist:
    in_vars.append('volumetric_soil_water_layer_1')
    print("EAR5 soil moisture is added")
        


dm = cl.data.IterDataModule(
    "downscaling",
    args.lowresdir, #"/lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg", 
    args.highresdir, #"/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg",
    in_vars,
    out_vars=out_vars, #[out_var_dict[args.variable]],
    subsample=1,
    batch_size=16, #32,
    buffer_size=500,
    num_workers=1,
)

dm.setup()

# Depricated in native_pytorch version
#print("dm.hparams",dm.hparams,flush=True)


# Set up deep learning model
# deprecated in lightning version
#model = cl.load_downscaling_module(device, data_module=dm, architecture="res_slimvit", train_loss='mse')
# new native pytorch version
#model, train_loss,val_losses,test_losses,train_transform,val_transforms,test_transforms = cl.load_downscaling_module(device,data_module=dm, architecture=args.preset, train_loss="mse")
# interim Feb/28~
model, train_loss,val_losses,test_losses,train_transform,val_transforms,test_transforms = cl.load_downscaling_module(device,data_module=dm, architecture=args.preset, train_loss='mse', train_target_transform=None, model_kwargs=model_kwargs)

model = model.to(device)
#model.update_spatial_resolution(spatial_resolution[data_key]) 
print('MODEL HERE')
print(model, flush=True)

#denorm = model.test_target_transforms[0]
denorm = test_transforms[0]
print("denorm is ",denorm,flush=True)


# this should be replaced
#model = cl.LitModule.load_from_checkpoint(
#    args.checkpoint_path,
#    net=model.net,
#    optimizer=model.optimizer,
#    lr_scheduler=None,
#    train_loss=None,
#    val_loss=None,
#    test_loss=model.test_loss,
#    test_target_tranfsorms=model.test_target_transforms,
#    map_location=device
#)

if os.path.exists(args.checkpoint_path):
    print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

    print("rank",dist.get_rank(),"src_rank",world_rank,flush=True)

    map_location = 'cuda:'+str(device)
    #map_location = 'cpu'

    checkpoint = torch.load(args.checkpoint_path,map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint

else:
    print("the checkpoint path does not exist.",flush=True)

    sys.exit("checkpoint path does not exist")

model = model.to(device)

# Setup trainer
#pl.seed_everything(0)
early_stopping = "train/perceptual:aggregate"

gpu_stats = DeviceStatsMonitor()



auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
            Block  # < ---- Your Transformer layer class
    },
)

#strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,activation_checkpointing=Block)



#trainer = pl.Trainer(
#    accelerator="gpu",
#    devices= world_size,
#    num_nodes = num_nodes,
#    max_epochs=1,
#    strategy=strategy,
#    precision="16",
#)
# Set the model to evaluation mode
model.eval()


#cl.utils.inference.test_on_single_image(
cl.utils.inference.test_on_many_images(
    model,
    dm,
    in_variables = in_vars,
    out_variables = out_vars, 
    in_transform=denorm,
    out_transform=denorm,
    variable=args.variable, #out_var_dict[args.variable],
    src="era5",
    outputdir=args.outputdir,
    device=device,
    index=-1
)

dist.destroy_process_group()
