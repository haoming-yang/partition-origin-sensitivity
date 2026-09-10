__all__ = ['PatchTST_backbone']


from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torch.nn.init import trunc_normal_

from layers.patchtst_layers import *
from layers.revin import RevIN
import math

class PatchTST_backbone(nn.Module):
    def __init__(self,configs, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):

        super().__init__()


        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)


        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        print(patch_num)

        self.backbone = TSTiEncoder(configs,c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)


        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)


    def forward(self, z):

        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)


        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0,1,3,2)


        z = self.backbone(z)
        z = self.head(z)


        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x




class TSTiEncoder(nn.Module):
    def __init__(self,configs, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):


        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len


        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len


        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)


        self.dropout = nn.Dropout(dropout)


        self.encoder = TSTEncoder(configs,q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)


    def forward(self, x) -> Tensor:

        n_vars = x.shape[1]

        x = x.permute(0,1,3,2)
        x = self.W_P(x)

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        u = self.dropout(u + self.W_pos)


        z = self.encoder(u)
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))
        z = z.permute(0,1,3,2)

        return z




class TSTEncoder(nn.Module):
    def __init__(self, configs, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        window_size = configs.window_size
        shift_size = math.ceil(window_size/2)
        PM = False
        self.layers = nn.ModuleList()
        for i in range(n_layers):

            self.layers.add(TSTEncoderLayer(configs,
                                                    window_size
                                                     ,
                                                     0 if (i % 2 == 0) else shift_size
                                                     ,
                                                      q_len
                                                     ,
                                                     d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn))
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)


        self.norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(2*dim), Transpose(1,2))

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, L, C).
        """
        B, L, C = x.shape


        pad_input = (L % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, L % 2))

        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], -1)

        x = self.norm(x)
        x = self.reduction(x)

        return x
def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    return windows

def compute_mask(L, window_size, shift_size):
    Lp = int(np.ceil(L / window_size)) * window_size
    img_mask = torch.zeros((1, Lp, 1))
    pad_size = int(Lp - L)
    if (pad_size == 0) or (pad_size + shift_size == window_size):
        segs = (slice(-window_size), slice(-window_size, -
                shift_size), slice(-shift_size, None))
    elif pad_size + shift_size > window_size:
        seg1 = int(window_size * 2 - L + shift_size)
        segs = (slice(-seg1), slice(-seg1, -window_size),
                slice(-window_size, -shift_size), slice(-shift_size, None))
    elif pad_size + shift_size < window_size:
        seg1 = int(window_size * 2 - L + shift_size)
        segs = (slice(-window_size), slice(-window_size, -seg1),
                slice(-seg1, -shift_size), slice(-shift_size, None))
    cnt = 0
    for d in segs:
        img_mask[:, d, :] = cnt
        cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(
        attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
class TSTEncoderLayer(nn.Module):
    def __init__(self, configs, window_size,shift_size, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.shift_size = shift_size
        if self.shift_size!=0:
            shift_size = int(window_size//2)
        if self.shift_size>0:
            self.mask = compute_mask(q_len, window_size, shift_size)

        self.res_attention = res_attention


        self.self_attn = WindowAttention1D(configs,window_size,shift_size,dim=d_model, num_heads=n_heads, qkv_bias=True, qk_scale=None, attn_drop=attn_dropout, proj_drop=dropout)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)


        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))


        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:


        if self.pre_norm:
            src = self.norm_attn(src)







        if self.shift_size>0:
            src2 = self.self_attn(src,self.mask.to(src.device))
        else:
            src2 = self.self_attn(src)

        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)


        if self.pre_norm:
            src = self.norm_ffn(src)

        src2 = self.ff(src)

        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)





        return src
















































































































class WindowAttention1D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, configs,window_size,shift_size, dim,  num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.shift_size = shift_size
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1), num_heads))


        coords_l = torch.arange(self.window_size)


        coords = torch.stack(torch.meshgrid(
            [coords_l]))


        print(coords)

        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - \
            1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def window_partition(self,x):
            """
            Args:
                x: (B, L, C)
                window_size (int): window size

            Returns:
                windows: (num_windows*B, window_size, C)
            """

            B, L, C = x.shape
            x = x.view(B, L // self.window_size, self.window_size, C)
            windows = x.permute(0, 1, 2, 3).contiguous().view(-1, self.window_size, C)
            return windows

    def window_reverse(self,windows, window_size, L):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            L (int): Sequence length

        Returns:
            x: (B, L, C)
        """
        B = int(windows.shape[0] / (L / window_size))
        x = windows.view(B, L // window_size, window_size, -1)
        x = x.permute(0, 1, 2, 3).contiguous().view(B, L, -1)
        return x
    def pad_shift_x(self,x,shift_size):
        B, L, C = x.shape

        pad_l = 0
        pad_r = (self.window_size - L % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r))
        _, Lp, _ = x.shape
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=-shift_size, dims=(1))

        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x)
        return x_windows,Lp

    def forward(self, x, mask=None):

        B, L, C = x.shape
        shift_size = self.shift_size
        x,Lp = self.pad_shift_x(x,shift_size)
        x = self.window_partition(x)
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wl, Wl) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv=qkv.reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale


        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()


        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(-1, *(self.window_size, C))
        x = self.window_reverse(x, self.window_size, Lp)
        if shift_size > 0:
            x = torch.roll(x, shifts=shift_size, dims=(1))
        else:
            x = x
        x = x[:, :L, :].contiguous()

        return x
