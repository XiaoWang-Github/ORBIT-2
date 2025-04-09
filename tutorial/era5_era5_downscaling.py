# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
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
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock
)



from pytorch_lightning.callbacks import DeviceStatsMonitor
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
os.environ['MASTER_PORT'] = "29500"
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
os.environ['RANK'] = os.environ['SLURM_PROCID']

world_size = int(os.environ['SLURM_NTASKS'])
world_rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])

num_nodes = world_size//8

torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

if world_rank==0:
    print("world_size",world_size,"num_nodes",num_nodes,flush=True)


parser = ArgumentParser()
parser.add_argument("era5_low_res_dir")
parser.add_argument("era5_high_res_dir")
parser.add_argument("preset", choices=["resnet", "unet", "vit","res_slimvit"])
parser.add_argument(
    "variable", choices=["t2m", "z500", "t850","u10","tp"], help="The variable to predict."
)
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

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
    args.era5_low_res_dir,
    args.era5_high_res_dir,
    in_vars,
    out_vars=[out_var_dict[args.variable]],
    subsample=1,
    batch_size=8,
    buffer_size=200,
    num_workers=2,
)
dm.setup()

# Set up deep learning model
model = cl.load_downscaling_module(device,data_module=dm, architecture=args.preset)


if world_rank==0:
    print("print model here",model,flush=True)


# Setup trainer
pl.seed_everything(0)
default_root_dir = f"{args.preset}_downscaling_{args.variable}"
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")

gpu_stats = DeviceStatsMonitor()

callbacks = [
    RichModelSummary(max_depth=args.summary_depth),
    gpu_stats,
    ModelCheckpoint(
        dirpath=f"{default_root_dir}/checkpoints",
        filename="epoch_{epoch:03d}",
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,  # Save every epoch
        auto_insert_metric_name=False,
    ),
]

if args.preset =="vit" or args.preset=="res_slimvit":
   
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block  # < ---- Your Transformer layer class
        },
    )

    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,activation_checkpointing=Block)

elif args.preset =="unet":
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            UpBlock,DownBlock,MiddleBlock  # < ---- Your Transformer layer class
        },
    )

    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,activation_checkpointing=[UpBlock,DownBlock,MiddleBlock])
else: #resnet
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            ResidualBlock  # < ---- Your Transformer layer class
        },
    )

    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,activation_checkpointing=ResidualBlock)



trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    default_root_dir=default_root_dir,
    accelerator="gpu",
    devices= 8,
    num_nodes = num_nodes,
    max_epochs=args.max_epochs,
    strategy=strategy,
    precision="bf16-mixed",
)

# Train and evaluate model from scratch
if args.checkpoint is None:
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
# Resume training from saved model checkpoint
else:
    trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoint)
    trainer.test(model, datamodule=dm)

# Evaluate the model alone
#    model = cl.LitModule.load_from_checkpoint(
#        args.checkpoint,
#        net=model.net,
#        optimizer=model.optimizer,
#        lr_scheduler=None,
#        train_loss=None,
#        val_loss=None,
#        test_loss=model.test_loss,
#        test_target_tranfsorms=model.test_target_transforms,
#    )

#    trainer.test(model, datamodule=dm)
