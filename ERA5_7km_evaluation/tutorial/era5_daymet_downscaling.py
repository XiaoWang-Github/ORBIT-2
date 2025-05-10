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
import random
import time
import numpy as np

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    SR_PRESSURE_LEVELS,
    CONSTANTS
)
from timm.models.vision_transformer import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock
)



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
    for k in list(pretrain_model.keys()):
        if "pos_embed" in k:
            print(f"Removing pos_embed")
            del pretrain_model[k]
        if "var_" in k:
            print(f"Removing var_embed, var_query and var_agg")
            del pretrain_model[k]
        if  "token_embeds" in k:
            print(f"Removing token_embed")
            del pretrain_model[k]
        if "channel" in k:
            print("k:", k)
            pretrain_model[k.replace("channel", "var")] = pretrain_model[k]
            del pretrain_model[k]
    for k in list(pretrain_model.keys()):  #in pre-train model weights, but not fine-tuning model
        if k not in state_dict.keys():
            print(f"Removing key {k} from pretrained checkpoint: no exist")
            del pretrain_model[k]
        elif pretrain_model[k].shape != state_dict[k].shape:  #if pre-train and fine-tune model weights dimension doesn't match
            print(f"Removing key {k} from pretrained checkpoint: no matching shape", pretrain_model[k].shape, state_dict[k].shape)
            del pretrain_model[k]


  
#    for k in list( checkpoint_model.keys()):
#        print("after deletion. Name ",k,flush=True)

    # load pre-trained model
    msg = model.load_state_dict(pretrain_model, strict=False)
    print(msg)
    del pretrain_model



def replace_constant(y, yhat, out_variables):
    for i in range(yhat.shape[1]):
        # if constant replace with ground-truth value
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat


def training_step(
    batch,
    batch_idx,
    net,
    device: int,
    train_loss_metric,
    train_target_transform) -> torch.Tensor:
    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
    #NOTE debug memo: here check distibution of output precip if log-transformed or not
    #np.save("./test_y_LogTF.npy", y.detach().cpu().numpy())
        
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
    batch,
    batch_idx: int,
    net,
    device: int,
    val_loss_metrics,
    val_target_transforms) -> torch.Tensor:

    return evaluate_func(batch, "val",net,device,val_loss_metrics,val_target_transforms)


def evaluate_func(
    batch,
    stage: str,
    net,
    device: int,
    loss_metrics,
    target_transforms):

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




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(device):

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = dist.get_rank()
    local_rank = int(os.environ['SLURM_LOCALID'])


    print("world_size",world_size,"world_rank",world_rank,"local_rank",local_rank,flush=True)


    parser = ArgumentParser()
    parser.add_argument("era5_daymet_low_res_dir")
    parser.add_argument("era5_daymet_high_res_dir")
    parser.add_argument("preset", choices=["resnet", "unet", "vit","res_slimvit"])
    parser.add_argument(
        "variable", choices=["t2m", "z500", "t850","u10", 'tp', 'prcp'], help="The variable to predict."
    )
    parser.add_argument("--summary_depth", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--loss_function", choices=["mse","perceptual", "quantile", "imagegradient", 'masked_mse'], default='mse', help="The loss function to optimize network."
    )
    parser.add_argument("--pretrain", default=None)

    args = parser.parse_args()


    if world_rank==0:
        print("args is",args,flush=True)


    #if both checkpoint and pretrain are available, use checkpoint
    if args.checkpoint is not None and args.pretrain is not None:
        args.pretrain = None

    # Set up data
    # list of daymet ['land_sea_mask', 'latitude', 'orography', 'prcp', '2m_temperature', 'sea_surface_temperature', 'total_precipitation', 'geopotential_500', 'geopotential_850', 'u_component_of_wind_500', 'u_component_of_wind_850', 'v_component_of_wind_500', 'v_component_of_wind_850', 'temperature_500', 'temperature_850', 'specific_humidity_500', 'specific_humidity_850', 'days_of_year', 'time_of_day', 'hrs_each_step', 'num_steps_per_shard', 'extra_steps', 'lattitude']
    variables = [
        "land_sea_mask",
        "orography",
        "lattitude",
    #    'prcp',
    #    "2m_temperature",
        'sea_surface_temperature', 
    #    'total_precipitation',
    #    "10m_u_component_of_wind",
    #    "10m_v_component_of_wind",
        "geopotential",
        "temperature",
    #    "relative_humidity",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
    ]
    out_var_dict = {
#       "t2m": "2m_temperature",
#       "z500": "geopotential_500",
#       "t850": "temperature_850",
#       "u10": "10m_u_component_of_wind",
#       'tp': 'total_precipitation'  # era5 derived
        'prcp': 'prcp' # daymet derived
    }
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in SR_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    
    if world_rank==0:
        print("in_vars",in_vars,flush=True)

    #load data module
    data_module = cl.data.IterDataModule(
        "downscaling",
        args.era5_daymet_low_res_dir,
        args.era5_daymet_high_res_dir,
        in_vars,
        out_vars=[out_var_dict[args.variable]],
        subsample=1,
        batch_size=args.batch_size,
        buffer_size=400,
        num_workers=1,
    ).to(device)
    data_module.setup()

    # Set up deep learning model
    #train_target_transform = precip_transform if args.variable in ('tp' , 'prcp' ) else None
    model, train_loss,val_losses,test_losses,train_transform,val_transforms,test_transforms = cl.load_downscaling_module(device,data_module=data_module, architecture=args.preset, train_loss=args.loss_function, train_target_transform=None)
  
    if dist.get_rank()==0:
        print("train_loss",train_loss,"train_transform",train_transform,"val_losses",val_losses,"val_transforms",val_transforms,flush=True)
 
    model = model.to(device)


    #load model checkpoint
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            print("model resume from checkpoint",args.checkpoint," Checkpoint path found.",flush=True)

            #map_location = 'cuda:'+str(device)
            map_location = 'cpu'

            checkpoint = torch.load(args.checkpoint,map_location=map_location)
            model.load_state_dict(checkpoint['model_state_dict'])

            del checkpoint
        else:
            print("resume from checkpoint was set to True. But the checkpoint path does not exist.",flush=True)

            sys.exit("checkpoint path does not exist")

    #load pretrained model
    if args.pretrain is not None:
        if os.path.exists(args.pretrain):
            print("load pretrained model",args.pretrain," Pretrain path found.",flush=True)
            load_pretrained_weights(model,args.pretrain,device)  
        else:
            print("resume from pretrained model was set to True. But the pretrained model path does not exist.",flush=True)

            sys.exit("pretrain path does not exist")



    seed_everything(0)
    #default_root_dir = f"{args.preset}_downscaling_{args.variable}"
    default_root_dir = f"{args.preset}_downscaling_{args.variable}/{args.loss_function}"


    #set up layer wrapping
    if args.preset =="vit" or args.preset=="res_slimvit":
   
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Block, Sequential # < ---- Your Transformer layer class
            },
        )

        check_fn = lambda submodule: isinstance(submodule, Block)  or isinstance(submodule,Sequential)



    elif args.preset =="unet":
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                UpBlock,DownBlock,MiddleBlock  # < ---- Your Transformer layer class
            },
        )

        check_fn = lambda submodule: isinstance(submodule, UpBlock) or isinstance(submodule, DownBlock) or isinstance(submodule, MiddleBlock)



    else: #resnet
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                ResidualBlock  # < ---- Your Transformer layer class
            },
        )
        check_fn = lambda submodule: isinstance(submodule, ResidualBlock)



    #bfloat16 policy
    bfloatPolicy = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )



    #fully sharded FSDP

    model = FSDP(model, device_id = local_rank, process_group= None,sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD,auto_wrap_policy = auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False)



    #model = DDP(model, device_ids=[local_rank], output_device=[local_rank]) 

    #activation checkpointing
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )


    if torch.distributed.get_rank()==0:
        print("model is ",model,flush=True)




    #load optimzier and scheduler


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


    epoch_start = 0
    if args.checkpoint is not None:

        print("optimizer resume from checkpoint",args.checkpoint," Checkpoint path found.",flush=True)

        #map_location = 'cuda:'+str(device)
        map_location = 'cpu'

        checkpoint = torch.load(args.checkpoint,map_location=map_location)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']+1

        del checkpoint



    #get latitude and longitude
    lat, lon = data_module.get_lat_lon()

    # get train data loader
    train_dataloader = data_module.train_dataloader()

    # get validation data loader
    val_dataloader = data_module.val_dataloader()



    #perform training
    for epoch in range(epoch_start,args.max_epochs):
    
        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        loss = 0.0
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if world_rank==0:
            print("epoch ",epoch,flush=True)
    
        for batch_idx, batch in enumerate(train_dataloader):

            if world_rank==0:
                torch.cuda.synchronize(device=device)
                tic1 = time.perf_counter() 

            loss = training_step(batch, batch_idx,model,device,train_loss, train_target_transform=None)
            #exit(0)

            epoch_loss += loss.detach()
    
            if world_rank==0:
                print("epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," loss ",loss,flush=True)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
            if world_rank==0:
                if batch_idx % 10 == 0:
                    print("rank",world_rank,"batch_idx",batch_idx,"get_lr ",scheduler.get_lr(),"after optimizer step torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)


            if world_rank==0:
                torch.cuda.synchronize(device=device)
                tic4 = time.perf_counter() 
                print(f"my rank {dist.get_rank()}. tic4-tic1 in {(tic4-tic1):0.4f} seconds\n",flush=True)
    

        scheduler.step()
    
        if world_rank==0:
            epoch_loss = epoch_loss/(batch_idx+1)
            print("epoch: ",epoch," epoch_loss ",epoch_loss,flush=True)



        if world_rank ==0:    
            # Create checkpoint path
            os.makedirs(default_root_dir, exist_ok=True)
            checkpoint_path = f"{default_root_dir}/checkpoints"
            # Check whether the specified checkpointing path exists or not
            isExist = os.path.exists(checkpoint_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(checkpoint_path)
                print("The new checkpoint directory is created!")        


        print("rank",world_rank,"Before torch.save torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)


        model_states = model.state_dict()
        optimizer_states = optimizer.state_dict()
        scheduler_states = scheduler.state_dict()

        if world_rank == 0 :
     
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_states,
                'optimizer_state_dict': optimizer_states,
                'scheduler_state_dict': scheduler_states,
                }, checkpoint_path+"/"+"ERA5-Daymet"+"_rank_"+str(world_rank)+"_epoch_"+ str(epoch) +".ckpt")
     
        print("rank",world_rank,"After torch.save torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

        dist.barrier()
        del model_states
        del optimizer_states
        del scheduler_states


        #perform validation
        if epoch%2==0:
            with torch.no_grad():
                #tell the model that we are in eval mode. Matters because we have the dropout
                model.eval()

                if world_rank==0:
                    print("val epoch ",epoch,flush=True)
    
                for batch_idx, batch in enumerate(val_dataloader):

                    if world_rank==0:
                        torch.cuda.synchronize(device=device)
                        tic1 = time.perf_counter() 

                    losses = validation_step(batch, batch_idx,model,device,val_losses,val_transforms)
    
                    if world_rank==0:
                        print("val epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," losses ",losses,flush=True)
           
                        torch.cuda.synchronize(device=device)
                        tic4 = time.perf_counter() 
                        print(f"my rank {dist.get_rank()}. tic4-tic1 in {(tic4-tic1):0.4f} seconds\n",flush=True)
    


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


    print("Using dist.init_process_group. world_size ",world_size,flush=True)
    
    main(device)

    dist.destroy_process_group()
