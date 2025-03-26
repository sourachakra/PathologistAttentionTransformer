import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from timm.models.layers import trunc_normal_
from segm.model.blocks import Block
from segm.model.utils import init_weights, resize_pos_embed
from einops import rearrange

class Segmenter2(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        grd
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.grid_size = grd

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):

        H, W = self.grid_size,self.grid_size # 2x - 10, 4x - 20, 10x - 50 (32), 20x - 100 (36)
        
        x = self.encoder(im, return_features=True)
        #print('encoded feats:',x.shape[0],x.shape[1],x.shape[2])
        x2 = torch.reshape(x, (x.shape[0],int(np.sqrt(x.shape[1])),int(np.sqrt(x.shape[1])),x.shape[2]))
        #print('encoded feats:',x2.shape)
        #print('----')
        masks = self.decoder(x, (H, W))
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")

        return masks,x2

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


class VisionTransformer2(nn.Module):
    def __init__(
        self,
        grid_size,
        image_size = 384,
        patch_size = 8,
        n_layers = 12,
        d_model = 384,
        d_ff = 384*4,
        n_heads = 8,
        n_cls = 1,
        dropout = 0.1,
        drop_path_rate = 0.0,
        distilled = False,
        channels = 384,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        self.pos_embed = nn.Parameter(
            torch.randn(1, d_model, grid_size,grid_size)
        )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)
        trunc_normal_(self.pos_embed, std=0.02)
        self.pre_logits = nn.Identity()
        self.apply(init_weights)


    def forward(self, im, return_features=False):
        #print('ii:',im.shape)
        B, _, H, W = im.shape
        PS = self.patch_size

        x = im

        pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.dropout(x)

        x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
        x = torch.permute(x,(0,2,1))
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        return x

class MaskTransformer2(nn.Module):
    def __init__(
        self,
        n_cls = 1,
        patch_size= 1,
        d_encoder = 384,
        n_layers = 8,
        n_heads = 8, #32
        d_model = 1,
        d_ff = 4*384,
        drop_path_rate = 0,
        dropout = 0,
    ):
        super().__init__()
        # self.d_encoder = d_encoder
        self.patch_size = patch_size
        # self.n_layers = n_layers
        # self.n_cls = n_cls
        # self.d_model = d_model
        # self.d_ff = d_ff
        # self.scale = d_model ** -0.5

        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.apply(init_weights)


    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        masks = self.proj_dec(x)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        masks = F.sigmoid(masks)
        return masks

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_encoder = nn.Conv2d(2048, 1, 1) #resnet50 - 2048, dino - 384
        #self.d_encoder2 = nn.Conv2d(512, 1, 1)
        #self.relu = nn.Sigmoid()

    def forward(self, x):
        masks = self.d_encoder(x)
        return masks
        
def make_model(grid_size1):
    #model = Model2()
    encoder = VisionTransformer2(grid_size = grid_size1)
    decoder = MaskTransformer2()
    model = Segmenter2(encoder, decoder, 2,grd = grid_size1)
    model.cuda()

    return model

