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
import yaml

# Third party
import climate_learn as cl
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
    train_loss_metric) -> torch.Tensor:
    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
        
    yhat = net.forward(x,in_variables)
    yhat = replace_constant(y, yhat, out_variables)

    if y.size(dim=2)!=yhat.size(dim=2) or y.size(dim=3)!=yhat.size(dim=3):
        losses = train_loss_metric(yhat, y[:,:,0:yhat.size(dim=2),0:yhat.size(dim=3)])
    else:

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
 
    yhat = net.forward(x, in_variables)
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

        if y_.size(dim=2)!=yhat_.size(dim=2) or y_.size(dim=3)!=yhat_.size(dim=3):
            losses = lf(yhat_, y_[:,:,0:yhat_.size(dim=2),0:yhat_.size(dim=3)])
        else:
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
    low_res_dir = conf['data']['low_res_dir']
    high_res_dir = conf['data']['high_res_dir']
    preset = conf['model']['preset']
    dict_out_variables = conf['data']['dict_out_variables']
    dict_in_variables = conf['data']['dict_in_variables']
    default_vars =  conf['data']['default_vars']
    spatial_resolution = conf['data']['spatial_resolution']

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

    if world_rank==0:
        print("max_epochs",max_epochs," ",checkpoint_path," ",pretrain_path," ",low_res_dir," ",high_res_dir,"spatial_resolution",spatial_resolution,"default_vars",default_vars,"preset",preset,"lr",lr,"beta_1",beta_1,"beta_2",beta_2,"weight_decay",weight_decay,"warmup_epochs",warmup_epochs,"warmup_start_lr",warmup_start_lr,"eta_min",eta_min,"superres_mag",superres_mag,"cnn_ratio",cnn_ratio,"patch_size",patch_size,"embed_dim",embed_dim,"depth",depth,"decoder_depth",decoder_depth,"num_heads",num_heads,"mlp_ratio",mlp_ratio,"drop_path",drop_path,"drop_rate",drop_rate,"batch_size",batch_size,"num_workers",num_workers,"buffer_size",buffer_size,flush=True)


    model_kwargs = {'default_vars':default_vars,'superres_mag':superres_mag,'cnn_ratio':cnn_ratio,'patch_size':patch_size,'embed_dim':embed_dim,'depth':depth,'decoder_depth':decoder_depth,'num_heads':num_heads,'mlp_ratio':mlp_ratio,'drop_path':drop_path,'drop_rate':drop_rate}


    if world_rank==0:
        print("model_kwargs",model_kwargs,flush=True)


    if preset!="vit" and preset!="res_slimvit":
        print("Only supports vit or residual slim vit training.",flush=True)
        sys.exit("Not vit or res_slimvit architecture")


    #if both checkpoint and pretrain are available, use checkpoint
    if checkpoint_path is not None and pretrain_path is not None:
        pretrain_path = None

    # Set up data

    data_key = "ERA5_2"


    in_temp = dict_in_variables[data_key]
    in_vars = []
    
    for var in in_temp:
        if var in PRESSURE_LEVEL_VARS:
            default_vars.remove(var)
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
                default_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)

    out_temp = dict_out_variables[data_key]
    out_vars = []

    for var in out_temp:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)

    
    

    if world_rank==0:
        print("in_vars",in_vars,flush=True)
        print("out_vars",out_vars,flush=True)
        print("updated default_vars",default_vars,flush=True)

    #load data module
    data_module = cl.data.IterDataModule(
        "downscaling",
        low_res_dir[data_key],
        high_res_dir[data_key],
        in_vars,
        out_vars=out_vars,
        subsample=1,
        batch_size=batch_size,
        buffer_size=buffer_size,
        num_workers=num_workers,
    ).to(device)
    data_module.setup()

    # Set up deep learning model
    model, train_loss,val_losses,test_losses,train_transform,val_transforms,test_transforms = cl.load_downscaling_module(device,data_module=data_module, architecture=preset,model_kwargs=model_kwargs)
  
    if dist.get_rank()==0:
        print("train_loss",train_loss,"train_transform",train_transform,"val_losses",val_losses,"val_transforms",val_transforms,flush=True)

    model = model.to(device)


    model.update_spatial_resolution(spatial_resolution[data_key]) 



    #load model checkpoint
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print("model resume from checkpoint",checkpoint_path," Checkpoint path found.",flush=True)

            #map_location = 'cuda:'+str(device)
            map_location = 'cpu'

            checkpoint = torch.load(checkpoint_path,map_location=map_location)
            model.load_state_dict(checkpoint['model_state_dict'])

            del checkpoint
        else:
            print("resume from checkpoint was set to True. But the checkpoint path does not exist.",flush=True)

            sys.exit("checkpoint path does not exist")

    #load pretrained model
    if pretrain_path is not None:
        if os.path.exists(pretrain_path):
            print("load pretrained model",pretrain_path," Pretrain path found.",flush=True)
            load_pretrained_weights(model,pretrain_path,device)  
        else:
            print("resume from pretrained model was set to True. But the pretrained model path does not exist.",flush=True)

            sys.exit("pretrain path does not exist")



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


    epoch_start = 0
    if checkpoint_path is not None:

        print("optimizer resume from checkpoint",checkpoint_path," Checkpoint path found.",flush=True)

        #map_location = 'cuda:'+str(device)
        map_location = 'cpu'

        checkpoint = torch.load(checkpoint_path,map_location=map_location)
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
    for epoch in range(epoch_start,max_epochs):
    
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

            loss = training_step(batch, batch_idx,model,device,train_loss)

            epoch_loss += loss.detach()
    
            if world_rank==0:
                print("epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," loss ",loss,flush=True)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
            if world_rank==0:
                print("rank",world_rank,"batch_idx",batch_idx,"get_lr ",scheduler.get_lr(),"after optimizer step torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)


            if world_rank==0:
                torch.cuda.synchronize(device=device)
                tic4 = time.perf_counter() 
                print(f"my rank {dist.get_rank()}. tic4-tic1 in {(tic4-tic1):0.4f} seconds\n",flush=True)
    

        scheduler.step()
    
        if world_rank==0:
            print("epoch: ",epoch," epoch_loss ",epoch_loss,flush=True)



        if world_rank ==0:    
            checkpoint_path = "/lustre/orion/nro108/scratch/xf9/checkpoints/climate" 
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
                }, checkpoint_path+"/"+"interm"+"_rank_"+str(world_rank)+"_epoch_"+ str(epoch) +".ckpt")
     
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
