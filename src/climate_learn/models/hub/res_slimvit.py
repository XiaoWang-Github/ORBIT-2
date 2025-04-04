from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register
import torch
import torch.nn as nn
from functools import lru_cache
import numpy as np
import torch.distributed as dist
# Third party
from timm.models.vision_transformer import trunc_normal_
from .components.attention import VariableMapping_Attention
from einops import rearrange
from .components.pos_embed import interpolate_pos_embed_on_the_fly, interpolate_pos_embed_on_the_fly_adaptive
from .components.patch_embed import PatchEmbed 
from .components.vit_blocks import Block
from climate_learn.utils.dist_functions import F_Identity_B_Broadcast, Grad_Inspect
from climate_learn.utils.fused_attn import FusedAttn
import kornia.filters as K
import random
from .components.adaptive_patching import Patchify


@register("res_slimvit")
class Res_Slim_ViT(nn.Module):
    def __init__(
        self,
        default_vars,  #list of default variables to be used for training
        img_size,
        in_channels,
        out_channels,
        history,
        superres_mag = 4,
        cnn_ratio = 4,
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        tensor_par_size = 1,
        tensor_par_group = None,
        FusedAttn_option = FusedAttn.CK,
        adaptive_patching=False,
        fixed_length=1024,
        smooth=[1,3,5],
        canny_tol=[.1,.2],
    ):
        super().__init__()
        self.default_vars = default_vars


        self.img_size = img_size
        self.cnn_ratio = cnn_ratio
        self.superres_mag = superres_mag
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.history = history
        self.embed_dim = embed_dim
        self.spatial_resolution = 0
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group


        self.spatial_embed = nn.Linear(1, embed_dim)

        self.adaptive_patching = adaptive_patching
        self.fixed_length = fixed_length
        self.smooth = smooth
        self.canny_tol = canny_tol
        
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
        )
        if self.adaptive_patching:
            self.num_patches = fixed_length
            self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=1)
            self.to_img = nn.Linear(embed_dim,patch_size**2)
            self.to_emb = nn.Linear(patch_size**2,embed_dim)
            self.magnify = nn.Linear(self.out_channels*patch_size**2, self.out_channels*(patch_size*superres_mag)**2)
        else:
            self.num_patches = self.token_embeds[0].num_patche

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables

        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        #self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.var_agg = VariableMapping_Attention(embed_dim, fused_attn=FusedAttn_option, num_heads=num_heads, qkv_bias=False,tensor_par_size = tensor_par_size, tensor_par_group = tensor_par_group)
        
        if self.adaptive_patching:
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * .02, requires_grad=learn_pos_emb
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
            )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads =num_heads, 
                    fused_attn=FusedAttn_option,
                    mlp_ratio = mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                    tensor_par_size = tensor_par_size,
                    tensor_par_group = tensor_par_group,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        #skip connection path
        self.path2 = nn.ModuleList()
        self.path2.append(nn.Conv2d(in_channels=(out_channels+4), out_channels=cnn_ratio*superres_mag*superres_mag, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path2.append(nn.GELU())
        self.path2.append(nn.PixelShuffle(superres_mag))
        self.path2.append(nn.Conv2d(in_channels=cnn_ratio, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path2 = nn.Sequential(*self.path2)


        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim,out_channels * (superres_mag*patch_size)**2))
        self.head = nn.Sequential(*self.head)
       
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1) 
        self.initialize_weights()

    def initialize_weights(self):
        if not self.adaptive_patching:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.img_size[0] // self.patch_size,
                self.img_size[1] // self.patch_size,
                cls_token=False,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)




    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def data_config(self, res, img_size, in_channels, out_channels, fixed_length, smooth, canny_tol):
        with torch.no_grad(): 
            orig_size = self.img_size

            self.spatial_resolution = res
            self.img_size = img_size
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.smooth = smooth
            self.canny_tol = canny_tol
            if self.adaptive_patching:
                self.num_patches = fixed_length
                self.patchify = Patchify(fixed_length=fixed_length, patch_size=self.patch_size, num_channels=1)
            else:
                self.num_patches = img_size[0] * img_size[1]// (self.patch_size **2)
       
 
        if torch.distributed.get_rank()==0:
            print("updated res is ",res,"img_size",img_size,"in_channels",in_channels,"out_channels",out_channels,"num_patches",self.num_patches,flush=True)


        if torch.distributed.get_rank()==0:
            print("model.pos_embed.shape",self.pos_embed.shape,flush=True)


    def unpatchify(self, x: torch.Tensor, scaling =1, out_channels=1):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = out_channels
        h = self.img_size[0] * scaling // p
        w = self.img_size[1] *scaling // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)


    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]


    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map



    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape

        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        #var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)

        var_query = self.var_query.expand(x.shape[0], -1, -1).contiguous()

        #x , _ = self.var_agg(var_query, x, x)
        x = self.var_agg(var_query, x)  # BxL, V~ , D, where V~ is the aggregated variables

        x = x.squeeze()

        if self.tensor_par_size >1:

            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)  #must do the backward broadcast because of the randomneess of dropout

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, V~, D

        return x

    def deserialize(self, x: torch.Tensor, qdt_list, out_channels=1, scaling=1):
        B = x.shape[0]
        #c = self.out_channels
        
                    
        x_list = []
        for i in range(B):
            x_list.append(torch.from_numpy(qdt_list[i].deserialize(np.expand_dims(x[i].to(torch.float32).detach().cpu().numpy(), axis=-1), self.patch_size*scaling, out_channels)).to(x.dtype).to(x.device))
            #x_list.append(qdt_list[i].deserialize(torch.unsqueeze(x[i], dim=-1), self.patch_size*scaling, out_channels).to(x.dtype))
        x = torch.stack([torch.moveaxis(x_list[i],-1,0) for i in range(len(x_list))])
        return x

    def residual_connection(self,x:torch.Tensor,out_var_index):
        """
         x: B, in channels, H, W
        """
        x = x[:,out_var_index,:,:]

        #x: B,out channels, H, W
        path2_result = self.path2(x)
        #x: B, output channels, H*mag, W*mag
        return path2_result


    def forward_encoder(self, x: torch.Tensor, variables):

        if isinstance(variables, list):
            variables = tuple(variables)

        #tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)

        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D, 

        # x.shape = [B,num_patches,embed_dim]

        if self.adaptive_patching:
            #Feature Space to Image Space
            x = self.to_img(x)
            # x.shape = [B,num_patches,patch_size*patch_size]
            x = self.unpatchify(x,scaling=1)
            # x.shape = [B,out_channels,h*patch_size, w*patch_size]

            smooth_factor = random.choice(self.smooth)
            xx = x.view(x.size(0), -1)
            #xx.shape = [B, out_channels*h*w]
            #Find max value in image per batch input
            xmin = torch.min(xx, dim=1)[0]
            #Find min value in image per batch input
            xmax = torch.max(xx, dim=1)[0]
            #Normalize to (0,1)
            xx = (xx-xmin.reshape(x.size(0),1,1))/(xmax.reshape(x.size(0),1,1)-xmin.reshape(x.size(0),1,1))
            if x.shape[0] > 1:
                xxx = torch.cat((xx[0][0],xx[1][1]))
                for i in range(x.shape[0]-2):
                    xxx = torch.cat((xxx,xx[i+2][i+2]))
                del xx
            else:
                xxx = xx[0][0]
                del xx
            xxx = xxx.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
            edges = K.canny(xxx.to(torch.float32)*255,sigma=(smooth_factor,smooth_factor),low_threshold=self.canny_tol[0],high_threshold=self.canny_tol[1])[1]
            del xxx

            B = x.shape[0]
            seq_img_list = []
            qdt_list = []
            for i in range(B):
                x_np = np.moveaxis(x[i].to(torch.float32).detach().cpu().numpy(), 0, -1)
                edges_np = edges[i][0].detach().cpu().numpy()
                seq_img, qdt = self.patchify(x_np, edges_np)
                #seq_img, qdt = self.patchify(x[i],edges[i][0])
                seq_img_list.append(seq_img)
                qdt_list.append(qdt)
            x = torch.from_numpy(np.stack([seq_img_list[k] for k in range(len(seq_img_list))])).to(x.dtype).to(x.device)
            #x = torch.stack([seq_img_list[k].to(x.dtype) for k in range(len(seq_img_list))])

            x = self.to_emb(x)
            pos_emb = interpolate_pos_embed_on_the_fly_adaptive(self.pos_embed,self.num_patches)

        else:
            qdt_list = None
            pos_emb = interpolate_pos_embed_on_the_fly(self.pos_embed,self.patch_size,self.img_size)


        x = x + pos_emb

        # add spatial resolution embedding

        spatial_emb = self.spatial_embed(torch.tensor(self.spatial_resolution,dtype=x.dtype,device=x.device).unsqueeze(-1))  # D

        spatial_emb = spatial_emb.unsqueeze(0).unsqueeze(0)  #1, 1, D

        x = x + spatial_emb  # B, L, D


        x = self.pos_drop(x)

        if self.tensor_par_size>1:
            src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
            dist.broadcast(x, src_rank , group=self.tensor_par_group)


        for blk in self.blocks:
            x = blk(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.norm(x)

        if self.tensor_par_size>1:
            x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

        return x,qdt_list

    
    def find_var_index(self,in_variables,out_variables):
        temp_index= [in_variables.index(variable) for variable in out_variables] 
        temp_index.append(in_variables.index("land_sea_mask"))
        temp_index.append(in_variables.index("orography"))
        temp_index.append(in_variables.index("lattitude"))
        temp_index.append(in_variables.index("landcover"))


        return temp_index

    def forward(self, x, in_variables,out_variables):
        if len(x.shape) == 5:  # x.shape = [B,T,in_channels,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]

        out_var_index = self.find_var_index(in_variables,out_variables)

        path2_result = self.residual_connection(x,out_var_index)     

        x,qdt_list = self.forward_encoder(x, in_variables)

        # x.shape = [B,num_patches,embed_dim]

        #decoder
        x = self.head(x) 

        if self.adaptive_patching:
            # x.shape = [B,fixed_length,out_channels*patch_size*patch_size]
            x = self.deserialize(x, qdt_list, out_channels=self.out_channels, scaling=self.superres_mag)
            #x = self.deserialize(x, qdt_list, out_channels=self.out_channels, scaling=1)
            # x.shape = [B,out_channels,h, w]
            x = x.reshape(shape=(x.shape[0], self.out_channels, self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size, self.patch_size, self.patch_size))
            x = torch.einsum("nchwpq->nhwpqc", x)
            x = x.reshape(shape=(x.shape[0], self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size, self.patch_size*self.patch_size*self.out_channels))
            # x.shape = [B,num_patches,out_channels*(patch_size)**2]
            x = self.magnify(x)
            # x.shape = [B,num_patches,out_channels*(patch_size*mag)**2]
            x = self.unpatchify(x,scaling=self.superres_mag, out_channels=self.out_channels)
            # x.shape = [B,out_channels,h*patch_size, w*patch_size]
        else:
            # x.shape = [B,num_patches,out_channels*patch_size*patch_size]
            x = self.unpatchify(x,scaling=self.superres_mag, out_channels=self.out_channels)
            # x.shape = [B,out_channels,h*patch_size, w*patch_size]

        x = self.conv_out(x) 
 
        if path2_result.size(dim=2) !=x.size(dim=2) or path2_result.size(dim=3) !=x.size(dim=3):
            preds = x + path2_result[:,:,0:x.size(dim=2),0:x.size(dim=3)]
        else:
            preds = x + path2_result

        return preds
