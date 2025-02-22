# Local application
from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register
from .adaptive_patching import Patchify

# Third party
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
from einops import rearrange
from functools import lru_cache
import numpy as np

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
        adaptive_patching=False,
        separate_channels=False,
        fixed_length=4096,
    ):
        super().__init__()
        self.default_vars = default_vars


        self.img_size = img_size
        self.cnn_ratio = cnn_ratio
        self.superres_mag = superres_mag
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.adaptive_patching = adaptive_patching
        self.separate_channels = separate_channels

        self.history = history

        if self.adaptive_patching:
            patch_dim = self.in_channels*self.patch_size**2
            patch_dim_woc = self.patch_size**2
            self.token_embeds = nn.ModuleList(
                #[nn.Linear(patch_dim, embed_dim) for i in range(len(default_vars))]
                [nn.Linear(patch_dim_woc, embed_dim) for i in range(len(default_vars))]
            )
            self.num_patches = fixed_length
            if self.adaptive_patching:
                if self.separate_channels:
                    #self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=1, sths=[self.gauss_filter_order])
                    self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=1)
                else:
                    #self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, sths=[self.gauss_filter_order])
                    self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=self.in_channels)
        else:
            self.token_embeds = nn.ModuleList(
                [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
            )
            self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables

        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

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
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        #skip connection path
        self.path2 = nn.ModuleList()
        self.path2.append(nn.Conv2d(in_channels=in_channels, out_channels=cnn_ratio*superres_mag*superres_mag, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path2.append(nn.GELU())
        self.path2.append(nn.PixelShuffle(superres_mag))
        self.path2.append(nn.Conv2d(in_channels=cnn_ratio, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path2 = nn.Sequential(*self.path2)


        #vit path
        self.path1 = nn.ModuleList()
        self.path1.append(nn.Conv2d(in_channels=out_channels, out_channels=cnn_ratio*superres_mag*superres_mag, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path1.append(nn.GELU())
        self.path1.append(nn.PixelShuffle(superres_mag))
        self.path1.append(nn.Conv2d(in_channels=cnn_ratio, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path1 = nn.Sequential(*self.path1)


        self.to_img = nn.Linear(embed_dim, out_channels * patch_size**2)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(self.img_size[1]//self.patch_size*self.patch_size*superres_mag, self.img_size[1]//self.patch_size*self.patch_size*superres_mag))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(self.img_size[1]//self.patch_size*self.patch_size*superres_mag, self.img_size[1]//self.patch_size*self.patch_size*superres_mag))
        self.head = nn.Sequential(*self.head)
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

    def unpatchify(self, x: torch.Tensor, scaling =1):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] * scaling // p
        w = self.img_size[1] *scaling // p
        assert h * w == x.shape[1]
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

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)

        #src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)

        x , _ = self.var_agg(var_query, x, x)
        #x = self.var_agg(var_query, x)  # BxL, V~ , D, where V~ is the aggregated variables

        x = x.squeeze()


#        x= F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)  #must do the backward broadcast because of the randomneess of dropout

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, V~, D


        return x

    def deserialize(self, x: torch.Tensor, qdt_list):
        B = x.shape[0]
        c = self.out_channels
        device = x.device


        x_list = []
        #print("X0_SHAPE", x[0].shape, flush=True)
        for i in range(B):
            if self.separate_channels:
                x_list.append(torch.from_numpy(qdt_list[i][0].deserialize(np.expand_dims(x[i].to(torch.float32).detach().cpu().numpy(), axis=-1), self.patch_size, c)).to(torch.bfloat16).to(device))
            else:
                x_list.append(torch.from_numpy(qdt_list[i].deserialize(np.expand_dims(x[i].to(torch.float32).detach().cpu().numpy(), axis=-1), self.patch_size, c)).to(torch.bfloat16).to(device))

        x = torch.stack([torch.moveaxis(x_list[i],-1,0) for i in range(len(x_list))])
        return x


    def forward_encoder(self, x: torch.Tensor, variables):
        device = x.device

        if isinstance(variables, list):
            variables = tuple(variables)

        #tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        
        if self.adaptive_patching:
            B = x.shape[0]
            C = x.shape[1]
            seq_img_list = []
            qdt_list = []
            for i in range(B):
                if self.separate_channels:
                    seq_img_channel_list = []
                    qdt_list.append([])
                    for j in range(C):
                        x_np = np.expand_dims(x[i,j].to(torch.float32).detach().cpu().numpy(), axis=-1)
                        seq_img, qdt = self.patchify(x_np)
                        seq_img_channel_list.append(seq_img)
                        qdt_list[i].append(qdt)
                    seq_img_list.append(np.stack([seq_image_channel_list[k] for k in range(len(seq_image_channel_list))]))
                else:
                    x_np = np.moveaxis(x[i].to(torch.float32).detach().cpu().numpy(), 0, -1)
                    seq_img, qdt = self.patchify(x_np)
                    seq_img_list.append(seq_img)
                    qdt_list.append(qdt)

            x = torch.from_numpy(np.stack([seq_img_list[k] for k in range(len(seq_img_list))])).to(torch.bfloat16).to(device)
            

        for i in range(len(var_ids)):
            id = var_ids[i]
            if self.adaptive_patching:
                embeds.append(self.token_embeds[id](torch.squeeze(x[:,i : i+1])))
            else:
                embeds.append(self.token_embeds[id](x[:,i : i+1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)

        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D, 

        # x.shape = [B,num_patches,embed_dim]

        #if torch.distributed.get_rank()==0:
        #    print("after patch_embed x.shape",x.shape,flush=True)


        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.norm(x)

        if not self.adaptive_patching:
            qdt_list = None

        return x, qdt_list

    def forward(self, x, in_variables):
        if len(x.shape) == 5:  # x.shape = [B,T,in_channels,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]

        path2_result = self.path2(x)
        
        x, qdt_list = self.forward_encoder(x, in_variables)

        # x.shape = [B,num_patches,embed_dim]

        x = self.to_img(x) 
        # x.shape = [B,num_patches,out_channels*patch_size*patch_size]
        if self.adaptive_patching:
            x = self.deserialize(x, qdt_list)
        else:
            x = self.unpatchify(x)
        # x.shape = [B,num_patches,h*patch_size, w*patch_size]
 
        x = self.path1(x)

        if path2_result.size(dim=2) !=x.size(dim=2) or path2_result.size(dim=3) !=x.size(dim=3):
            preds = x + path2_result[:,:,0:x.size(dim=2),0:x.size(dim=3)]
        else:
            preds = x + path2_result

        #decoder
        preds = self.head(preds) 
        # preds.shape = [B,out_channels,H,W]
        return preds
