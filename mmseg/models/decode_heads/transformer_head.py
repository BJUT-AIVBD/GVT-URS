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


class WindowAttention(nn.Module):
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
        coords_h = torch.arange(self.window_size_up)
        coords_w = torch.arange(self.window_size_up)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size_up - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size_up - 1
        relative_coords[:, :, 0] *= 2 * self.window_size_up - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(self.window_size ** 2, 3 * self.window_size_up ** 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x.transpose(1, 2)
        B_, C, N = x.shape
        qkv = self.qkv(x).reshape(B_, self.num_heads, C // self.num_heads, 3, 4 * N).permute(3, 0, 1, 4, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size_up * self.window_size_up, self.window_size_up * self.window_size_up,
            -1)  # 4*Wh*Ww,4*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, 4*Wh*Ww, 4*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, 4 * N, 4 * N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, 4 * N, 4 * N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, 4 * N, C)
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
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, H, W).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        x = self.norm(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # calculate mask matrix
            img_mask = torch.zeros((1, 2 * Hp, 2 * Wp, 1), device=x.device)  # 1 2*Hp 2*Wp 1
            h_slices = (slice(0, -self.window_size_up),
                        slice(-self.window_size_up, -self.shift_size_up),
                        slice(-self.shift_size_up, None))
            w_slices = (slice(0, -self.window_size_up),
                        slice(-self.window_size_up, -self.shift_size_up),
                        slice(-self.shift_size_up, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size_up)  # nW, 2*window_size, 2*window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size_up * self.window_size_up)
            mask_matrix = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            mask_matrix = mask_matrix.masked_fill(mask_matrix != 0, float(-100.0)).masked_fill(mask_matrix == 0,
                                                                                               float(0.0))

            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA based upsampling
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, upsample_rate*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size_up, self.window_size_up,
                                         C)  # nW*B, 2*window_size, 2*window_size, C
        shifted_x = window_reverse(attn_windows, self.window_size_up, Hp * self.upsample_rate,
                                   Wp * self.upsample_rate)  # B 2H' 2W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size_up, self.shift_size_up), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H * self.upsample_rate, :W * self.upsample_rate, :].contiguous()

        # FFN
        shortcut = resize(
            shortcut,
            size=[H * self.upsample_rate, W * self.upsample_rate],
            mode='bilinear',
            align_corners=False)

        shortcut = shortcut.permute(0, 2, 3, 1)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm(x)))

        x = x.permute(0, 3, 1, 2)

        return x


@HEADS.register_module()
class TransformerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:

    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(TransformerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        num_heads = [3, 6, 12, 24]
        window_size = 7
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

        # skip the top layer

        self.bottleneck_final = ConvModule(
            self.in_channels[0],
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.attn = nn.ModuleList()
        for i in range(1, len(self.in_channels)):
            layer = TransformerSampling(self.in_channels[i], num_heads[i], window_size)
            self.attn.append(layer)

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals

        for i in range(len(inputs) - 1, 0, -1):
            inputs[i] = self.attn[i - 1](inputs[i])
            inputs[i - 1] = self.bottlenecks[i - 1](inputs[i]) + inputs[i - 1]

        output = self.bottleneck_final(inputs[0])
        output = self.cls_seg(output)

        return output
