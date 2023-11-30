import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class RelativePositionBias(nn.Module):

    def __init__(self, channels, window_size, num_heads):

        super().__init__()
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = channels // num_heads

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)
    def forward(self, attn):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        return attn
class DSBlock(nn.Module):

    def __init__(self, Hs, Ws, in_channels, num_heads, conv_cfg, norm_cfg, act_cfg, align_corners, qkv_bias=True,
                 attn_drop=0.):
        super(DSBlock, self).__init__()
        self.Hs = Hs
        self.Ws = Ws
        self.channels = in_channels // 2
        self.num_heads = num_heads
        self.dep_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.l_conv = ConvModule(
            in_channels,
            in_channels,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.down_sample = nn.AdaptiveAvgPool2d((Hs, Ws))
        self.qkv = nn.Linear(self.channels, 3 * self.channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.align_corners = align_corners

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        Fl = self.dep_conv(inputs[:, :self.channels, :, :]).view(B, self.channels, H * W).transpose(1, 2)
        l_qkv = self.qkv(Fl).reshape(B, H * W, 3, self.num_heads, self.channels // self.num_heads).permute(2, 0, 3, 1,
                                                                                                           4)
        l_q, l_k, l_v = l_qkv[0], l_qkv[1], l_qkv[2]
        Fg = self.down_sample(inputs[:, self.channels:, :, :]).view(B, self.channels, self.Hs * self.Ws).transpose(1, 2)
        g_qkv = self.qkv(Fg).reshape(B, self.Hs * self.Ws, 3, self.num_heads, self.channels // self.num_heads).permute(
            2, 0, 3, 1, 4)
        g_q, g_k, g_v = g_qkv[0], g_qkv[1], g_qkv[2]

        attn_lg = (l_q @ g_k.transpose(-2, -1))
        attn_gl = (g_q @ l_k.transpose(-2, -1))

        attn_lg = self.softmax(attn_lg / (self.channels // self.num_heads) ** 0.5)
        attn_gl = self.softmax(attn_gl / (self.channels // self.num_heads) ** 0.5)

        attn_lg = self.attn_drop(attn_lg)
        attn_gl = self.attn_drop(attn_gl)

        Flg = (attn_lg @ g_v).transpose(1, 2).reshape(B, H, W, self.channels).permute(0, 3, 1, 2)
        Fgl = (attn_gl @ l_v).transpose(1, 2).reshape(B, self.Hs, self.Ws, self.channels).permute(0, 3, 1, 2)

        Fgl = resize(Fgl, size=(H, W), mode='bilinear', align_corners=self.align_corners)
        out = self.l_conv(torch.cat((Flg, Fgl), dim=1))

        return out


@HEADS.register_module()
class UPerHeadDualStreamFPN(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHeadDualStreamFPN, self).__init__(
            input_transform='multiple_select', **kwargs)
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
        self.fpn_convs = nn.ModuleList()
        self.ds_blocks = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            ds_block = DSBlock(7, 7, self.channels, 8, self.conv_cfg, self.norm_cfg, self.act_cfg, self.align_corners,
                               qkv_bias=True,
                               attn_drop=0.2)
            self.lateral_convs.append(l_conv)
            self.ds_blocks.append(ds_block)

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

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.ds_blocks[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output
