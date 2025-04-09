import climate_learn as cl
import torch
import os

from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import os
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


torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

num_nodes = 1
# assuming we are downscaling geopotential from ERA5

# Set up data
variables = [
    "land_sea_mask",
    "orography",
    "lattitude",
#    "toa_incident_solar_radiation",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential",
    "temperature",
#    "relative_humidity",
    "specific_humidity",
    "total_precipitation_24hr",
#    "u_component_of_wind",
#    "v_component_of_wind",
]
out_var_dict = {
    "t2m": "2m_temperature",
    "tp": "total_precipitation_24hr",
#    "z500": "geopotential_500",
#    "t850": "temperature_850",
#     "u10": "10m_u_component_of_wind"

}
in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)


dm = cl.data.IterDataModule(
    "downscaling",
    "/lustre/orion/lrn036/world-shared/data/superres/era5/5.625_deg", 
    "/lustre/orion/lrn036/world-shared/data/superres/era5/1.40625_deg",
    in_vars,
    out_vars=[out_var_dict["t2m"]],
    subsample=1,
    batch_size=32,
    buffer_size=300,
    num_workers=1,
)

dm.setup()

print("dm.hparams",dm.hparams,flush=True)


# Set up deep learning model
model = cl.load_downscaling_module(device, data_module=dm, architecture="unet")

model = model.to(device)

denorm = model.test_target_transforms[0]


model = cl.LitModule.load_from_checkpoint(
    "./reslim_downscaling_t2m/checkpoints/epoch_025.ckpt",
    net=model.net,
    optimizer=model.optimizer,
    lr_scheduler=None,
    train_loss=None,
    val_loss=None,
    test_loss=model.test_loss,
    test_target_tranfsorms=model.test_target_transforms,
    map_location=device
)

model = model.to(device)

# Setup trainer
pl.seed_everything(0)
early_stopping = "train/perceptual:aggregate"

gpu_stats = DeviceStatsMonitor()



auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
            Block  # < ---- Your Transformer layer class
    },
)

strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,activation_checkpointing=Block)



trainer = pl.Trainer(
    accelerator="gpu",
    devices= world_size,
    num_nodes = num_nodes,
    max_epochs=1,
    strategy=strategy,
    precision="16",
)



cl.utils.visualize.visualize_at_index(
    model,
    dm,
    in_transform=denorm,
    out_transform=denorm,
    variable="2m_temperature",
    src="era5",
    index=0  # visualize the first sample of the test set
)
