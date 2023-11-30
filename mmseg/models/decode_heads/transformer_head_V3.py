import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
# from tensorboardX import SummaryWriter


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CoWindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.window_size_up = 2 * window_size  # 2*Wh, 2*Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size_up - 1) * (2 * self.window_size_up - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        big_coords = torch.stack(torch.meshgrid([torch.arange(self.window_size_up), torch.arange(self.window_size_up)])) # 2, hb, wb
        big_coords_flatten = torch.flatten(big_coords, 1)  # 2, hb*wb

        small_coords = torch.stack(torch.meshgrid([torch.arange(self.window_size), torch.arange(self.window_size)])) # 2, hs, ws
        small_coords_flatten = torch.flatten(small_coords, 1)  # 2, hs*ws

        relative_coords = big_coords_flatten[:, :, None] - small_coords_flatten[:, None, :]  # 2, hb*wb, hs*ws
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # hb*wb, hs*ws, 2
        relative_coords[:, :, 0] += self.window_size_up - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size_up - 1
        relative_coords[:, :, 0] *= 2 * self.window_size_up - 1
        relative_position_index = relative_coords.sum(-1)  # hb*wb, hs*ws
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, big_x, small_x):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        Bb_, Nb, Cb = big_x.shape
        big_qkv = self.qkv(big_x).reshape(Bb_, Nb, 3, self.num_heads, Cb // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, Bb_, num_heads, Nb, Cb//num_heads
        qb, kb, vb = big_qkv[0], big_qkv[1], big_qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        Bs_, Ns, Cs = small_x.shape
        small_qkv = self.qkv(small_x).reshape(Bs_, Ns, 3, self.num_heads, Cs // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, Bs_, num_heads, Ns, Cs//num_heads
        qs, ks, vs = small_qkv[0], small_qkv[1], small_qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (qb @ ks.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size_up * self.window_size_up, self.window_size * self.window_size,
            -1)  # hb*wb, hs*ws,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, hb*wb, hs*ws

        attn = self.softmax(attn + relative_position_bias.unsqueeze(0))
        # attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ vs).transpose(1, 2).reshape(Bb_, Nb, Cs)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerSampling(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.upsample_rate = 2
        self.window_size_up = 2 * window_size
        self.shift_size = window_size // 2
        self.shift_size_up = window_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm = norm_layer(dim)
        self.CoAttn = CoWindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, big_x, small_x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, H, W).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = big_x

        # pad big feature map to multiples of window size
        big_x = big_x.permute(0, 2, 3, 1)
        Bb, Hb, Wb, Cb = big_x.shape
        big_x = self.norm(big_x)

        pad_bl = pad_bt = 0
        pad_br = (self.window_size_up - Wb % self.window_size_up) % self.window_size_up
        pad_bb = (self.window_size_up - Hb % self.window_size_up) % self.window_size_up
        big_x = F.pad(big_x, (0, 0, pad_bl, pad_br, pad_bt, pad_bb))
        _, Hbp, Wbp, _ = big_x.shape

        # pad small feature map to multiples of window size
        small_x = small_x.permute(0, 2, 3, 1)
        Bs, Hs, Ws, Cs = small_x.shape
        small_x = self.norm(small_x)
        assert Cs == Cb, 'channels between big feature maps and small feature maps must be same'

        pad_sl = pad_st = 0
        pad_sr = (self.window_size - Ws % self.window_size) % self.window_size
        pad_sb = (self.window_size - Hs % self.window_size) % self.window_size
        small_x = F.pad(small_x, (0, 0, pad_sl, pad_sr, pad_st, pad_sb))
        _, Hsp, Wsp, _ = small_x.shape

        # big feature map partition windows
        xb_windows = window_partition(big_x, self.window_size_up)  # nW*B, window_size_up, window_size_up, Cb
        xb_windows = xb_windows.view(-1, self.window_size_up * self.window_size_up,
                                     Cb)  # nW*B, window_size_up*window_size_up, Cb

        # small feature map partition windows
        xs_windows = window_partition(small_x, self.window_size)  # nW*B, window_size, window_size, Cs
        xs_windows = xs_windows.view(-1, self.window_size * self.window_size, Cs)  # nW*B, window_size*window_size, Cs

        # W-MSA based information fusion
        attn_windows = self.CoAttn(xb_windows, xs_windows)  # nW*B, window_size_up*window_size_up, Cs

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size_up, self.window_size_up,
                                         Cs)  # nW*B, window_size_up, window_size_up, Cs
        shifted_x = window_reverse(attn_windows, self.window_size_up, Hbp, Wbp)  # B 2H' 2W' C

        if pad_br > 0 or pad_bb > 0:
            shifted_x = shifted_x[:, :Hb, :Wb, :].contiguous()

        # FFN

        shortcut = shortcut.permute(0, 2, 3, 1)

        x = shortcut + self.drop_path(shifted_x)
        x = x + self.drop_path(self.mlp(self.norm(x)))

        x = x.permute(0, 3, 1, 2)

        return x


@HEADS.register_module()
class TransformerHeadV3(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:

    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(TransformerHeadV3, self).__init__(
            input_transform='multiple_select', **kwargs)
        num_heads = [8, 16, 32]
        window_size = 7
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        # bottleneck
        self.bottlenecks = nn.ModuleList()
        for i in range(1, len(self.in_channels)):
            layer = ConvModule(
                self.in_channels[i],
                self.in_channels[i - 1],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.bottlenecks.append(layer)

        self.attn = nn.ModuleList()
        for i in range(len(num_heads)):
            layer = TransformerSampling(self.channels, num_heads[i], window_size)
            self.attn.append(layer)

        self.conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            inplace=False)
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        used_backbone_levels = len(laterals)

        # writer = SummaryWriter('../submits/data/Swin-S/unpaved_TransformerHead_V2/Vis/')

        for i in range(used_backbone_levels - 1, 0, -1):
            # writer.add_images('before sampled {}'.format(i), laterals[i].permute(1, 0, 2, 3), 0)

            laterals[i - 1] = self.conv(self.attn[i - 1](laterals[i - 1], laterals[i]))

            # writer.add_images('after sampled {}'.format(i), laterals[i].permute(1, 0, 2, 3), 0)

        # writer.close()

        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i] = resize(
                laterals[i],
                size=laterals[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(laterals, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)

        return output
