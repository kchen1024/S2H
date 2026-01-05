import numbers
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import mmcv
from mmcv.runner import auto_fp16

from mmedit.models.base import BaseModel
from mmedit.models.registry import MODELS
from mmedit.models.builder import build_loss
from mmedit.core import psnr, ssim, tensor2img
from mmedit.utils import get_root_logger
import numpy as np
from ailut import ailut_transform


class BasicBlock(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class TPAMIBackbone(nn.Sequential):
    r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).

    Args:
        pretrained (bool, optional): [ignored].
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to insert an extra pooling layer
            at the very end of the module to reduce the number of parameters of
            the subsequent module. Default: False.
    """

    def __init__(self, pretrained=False, input_resolution=256, extra_pooling=False):
        body = [
            BasicBlock(3, 16, stride=2, norm=True),
            BasicBlock(16, 32, stride=2, norm=True),
            BasicBlock(32, 64, stride=2, norm=True),
            BasicBlock(64, 128, stride=2, norm=True),
            BasicBlock(128, 128, stride=2),
            nn.Dropout(p=0.5),
        ]
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = 128 * (4 if extra_pooling else 64)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


class Res18Backbone(nn.Module):
    r"""The ResNet-18 backbone.

    Args:
        pretrained (bool, optional): Whether to use the torchvison pretrained weights.
            Default: True.
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 224.
        extra_pooling (bool, optional): [ignore].
    """

    def __init__(self, pretrained=True, input_resolution=224, extra_pooling=False):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=True)
        return self.net(imgs).view(imgs.shape[0], -1)


class LUTGenerator(nn.Module):
    r"""The LUT generator module (mapping h).

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks in the mapping h (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity


class AdaInt(nn.Module):
    r"""The Adaptive Interval Learning (AdaInt) module (mapping g).

    It consists of a single fully-connected layer and some post-process operations.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        adaint_share (bool, optional): Whether to enable Share-AdaInt. Default: False.
    """

    def __init__(self, n_colors, n_vertices, n_feats, adaint_share=False) -> None:
        super().__init__()
        repeat_factor = n_colors if not adaint_share else 1
        self.intervals_generator = nn.Linear(
            n_feats, (n_vertices - 1) * repeat_factor)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.adaint_share = adaint_share

    def init_weights(self):
        r"""Init weights for models.

        We use all-zero and all-one initializations for its weights and bias, respectively.
        """
        nn.init.zeros_(self.intervals_generator.weight)
        nn.init.ones_(self.intervals_generator.bias)

    def forward(self, x):
        r"""Forward function for AdaInt module.

        Args:
            x (tensor): Input image representation, shape (b, f).
        Returns:
            Tensor: Sampling coordinates along each lattice dimension, shape (b, c, d).
        """
        x = x.view(x.shape[0], -1)
        intervals = self.intervals_generator(x).view(
            x.shape[0], -1, self.n_vertices - 1)
        if self.adaint_share:
            intervals = intervals.repeat_interleave(self.n_colors, dim=1)
        intervals = intervals.softmax(-1)
        vertices = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
        return vertices


class PreRefineHead(nn.Module):
    r"""LUT 前处理：平滑 SDR 输入的块状伪影和色阶分层
    
    在源头处理问题，避免 LUT 放大伪影。
    低分辨率处理 + 上采样产生平滑残差。
    
    设计思路：
    - 16x16 编码块和高亮分层都是低频问题
    - 下采样 4x 后处理，感受野覆盖多个块
    - 上采样产生自然平滑的残差
    - 只修正输入，不改变 LUT 的自适应能力
    
    4K 推理约 1-2ms

    Args:
        in_channels (int): Number of input channels. Default: 3.
        hidden (int): Hidden layer dimension. Default: 16.
        scale_factor (int): Downsampling factor. Default: 4.
        residual_scale (float): Scale for residual output. Default: 0.05.
    """

    def __init__(self, in_channels=3, hidden=16, scale_factor=4, residual_scale=0.05):
        super().__init__()
        self.scale_factor = scale_factor
        self.residual_scale = residual_scale

        # 低分辨率处理网络
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 3, 1)
        )

        self._init_weights()

    def _init_weights(self):
        r"""Initialize weights for near-identity mapping at start."""
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        r"""Forward function.
        
        Args:
            x (Tensor): Input SDR image, shape (B, 3, H, W).
        
        Returns:
            Tensor: Refined SDR image.
            Tensor: Residual for debugging/loss.
        """
        H, W = x.shape[2:]
        
        # 计算低分辨率尺寸 (确保是整数)
        H_low = H // self.scale_factor
        W_low = W // self.scale_factor

        # 下采样到低分辨率
        x_low = F.interpolate(x, size=(H_low, W_low),
                              mode='bilinear', align_corners=True)
        
        # 低分辨率残差预测
        res_low = torch.tanh(self.net(x_low)) * self.residual_scale
        
        # 上采样回原分辨率 (bilinear 产生平滑过渡)
        residual = F.interpolate(res_low, size=(H, W),
                                 mode='bilinear', align_corners=True)

        refined = x + residual
        
        return refined, residual


class SpatialOffsetGenerator(nn.Module):
    r"""空间自适应偏移生成器：让 LUT 具有空间感知能力
    
    支持两种模式：
    - 作用于输入 (offset_mode='input'): output = LUT(input + offset)
    - 作用于输出 (offset_mode='output'): output = LUT(input) + offset
    
    作用于输出的优点：
    - 不受 LUT 精细度影响，训练更稳定
    - offset 效果完全可控（线性叠加）
    - 没有边界限制问题

    Args:
        hidden (int): Hidden layer dimension. Default: 16.
        scale_factor (int): Downsampling factor. Default: 1.
        offset_scale (float): Maximum offset magnitude. Default: 0.1.
    """

    def __init__(self, hidden=16, scale_factor=1, offset_scale=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.offset_scale = offset_scale

        # 轻量级网络：输入 RGB，输出 3 通道偏移
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 3, 1)
        )

        self._init_weights()

    def _init_weights(self):
        r"""Initialize to near-zero offset at start."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        r"""Generate spatial-adaptive offset.
        
        Args:
            x (Tensor): Input image, shape (B, 3, H, W), range [0, 1].
        
        Returns:
            Tensor: Offset map, shape (B, 3, H, W), range [-offset_scale, +offset_scale].
        """
        H, W = x.shape[2:]

        if self.scale_factor == 1:
            # 全分辨率
            offset = torch.tanh(self.net(x)) * self.offset_scale
        else:
            # 低分辨率处理
            H_low = H // self.scale_factor
            W_low = W // self.scale_factor
            x_low = F.interpolate(x, size=(H_low, W_low),
                                  mode='bilinear', align_corners=True)
            offset_low = torch.tanh(self.net(x_low)) * self.offset_scale
            offset = F.interpolate(offset_low, size=(H, W),
                                   mode='bilinear', align_corners=True)

        # 零均值约束已移除 - 在 output 模式下会导致问题
        # offset = offset - offset.mean(dim=[2, 3], keepdim=True)

        return offset


class SpatialAdaptiveLUT(nn.Module):
    r"""空间自适应 LUT 模块
    
    支持两种模式：
    - offset_mode='input': output = LUT(input + offset)，修改查表坐标
    - offset_mode='output': output = LUT(input) + offset，直接修正输出（推荐）
    
    推荐使用 'output' 模式，因为：
    - 不受 LUT 精细度影响，训练更稳定
    - offset 效果完全可控
    - 没有边界问题
    
    Args:
        hidden (int): Hidden dimension for offset generator. Default: 16.
        offset_scale_factor (int): Downsampling for offset generator. Default: 1.
        offset_scale (float): Maximum offset magnitude. Default: 0.1.
        offset_mode (str): 'input' or 'output'. Default: 'output'.
    """

    def __init__(self, hidden=16, offset_scale_factor=1, offset_scale=0.1,
                 smooth_scale_factor=4, smooth_kernel_size=5, en_smooth=False,
                 offset_mode='output'):
        super().__init__()
        self.offset_mode = offset_mode
        
        # Spatial offset generator
        self.offset_gen = SpatialOffsetGenerator(
            hidden=hidden,
            scale_factor=offset_scale_factor,
            offset_scale=offset_scale
        )

    def forward(self, x, lut_out=None):
        r"""Process for spatial-adaptive LUT.
        
        Args:
            x (Tensor): Input SDR image, shape (B, 3, H, W), range [0, 1].
            lut_out (Tensor, optional): LUT output, only used for 'output' mode.
        
        Returns:
            If offset_mode='input':
                Tensor: Modified input for LUT lookup.
                Tensor: Spatial offset (for debugging).
                None: Placeholder.
            If offset_mode='output':
                Tensor: Spatial offset to add to LUT output.
        """
        # 生成 spatial offset（基于原始输入）
        offset = self.offset_gen(x)
        
        if self.offset_mode == 'input':
            # 作用于输入：修改查表坐标
            # 直接加 offset，然后 clamp 结果到 [0, 1]
            output = (x + offset).clamp(0, 1)
            return output, offset, None
        else:
            # 作用于输出：直接返回 offset
            return offset


class PostRefineHead(nn.Module):
    r"""LUT 后处理：修复 LUT 输出中被放大的伪影
    
    处理 LUT 变换后残留的块状伪影和色阶分层。
    结合 LUT 输出和原始 SDR 输入进行修正。
    
    设计思路：
    - LUT 会放大输入的伪影，后处理直接修复放大后的问题
    - 输入包含 LUT 输出 + SDR，提供更多上下文
    - 低分辨率处理保持速度
    
    4K 推理约 1-2ms

    Args:
        in_channels (int): Number of input channels (lut_out + sdr). Default: 6.
        hidden (int): Hidden layer dimension. Default: 16.
        scale_factor (int): Downsampling factor. Default: 4.
        residual_scale (float): Scale for residual output. Default: 0.1.
    """

    def __init__(self, in_channels=6, hidden=16, scale_factor=4, residual_scale=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.residual_scale = residual_scale

        # 低分辨率处理网络
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 3, 1)
        )

        self._init_weights()

    def _init_weights(self):
        r"""Initialize weights for near-identity mapping at start."""
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lut_out, sdr):
        r"""Forward function.
        
        Args:
            lut_out (Tensor): LUT output image, shape (B, 3, H, W).
            sdr (Tensor): Original SDR input, shape (B, 3, H, W).
        
        Returns:
            Tensor: Refined HDR image.
            Tensor: Residual for debugging/loss.
        """
        H, W = lut_out.shape[2:]
        
        # 拼接 LUT 输出和 SDR 输入
        x = torch.cat([lut_out, sdr], dim=1)  # (B, 6, H, W)
        
        if self.scale_factor == 1:
            # 全分辨率处理，直接卷积
            residual = torch.tanh(self.net(x)) * self.residual_scale
        else:
            # 低分辨率处理
            H_low = H // self.scale_factor
            W_low = W // self.scale_factor
            x_low = F.interpolate(x, size=(H_low, W_low),
                                  mode='bilinear', align_corners=True)
            res_low = torch.tanh(self.net(x_low)) * self.residual_scale
            residual = F.interpolate(res_low, size=(H, W),
                                     mode='bilinear', align_corners=True)

        refined = lut_out + residual
        
        return refined, residual


class LightweightDeblocking(nn.Module):
    r"""轻量级 Deblocking 后处理模块
    
    专门针对 16x16 块状伪影和色阶分层设计。
    
    设计特点：
    1. 全分辨率或 2x 下采样处理，保留块边界细节
    2. 感受野覆盖 16x16 块（通过 dilated conv）
    3. 残差学习：只学习修正量，任务更简单
    
    4K 推理约 3-6ms
    
    Args:
        hidden (int): Hidden layer dimension. Default: 24.
        scale_factor (int): Downsampling factor (1 or 2). Default: 2.
        residual_scale (float): Maximum residual magnitude. Default: 0.15.
    """

    def __init__(self, hidden=24, scale_factor=2, residual_scale=0.15):
        super().__init__()
        self.scale_factor = scale_factor
        self.residual_scale = residual_scale
        
        # 输入通道：LUT输出(3) + SDR(3)
        in_channels = 6
        
        # 多尺度感受野网络
        # 使用 dilated conv 扩大感受野覆盖 16x16 块
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2)  # 感受野 7x7
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4)  # 感受野 15x15
        self.conv4 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2)  # 再次 7x7
        self.conv_out = nn.Conv2d(hidden, 3, 1)
        
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        self._init_weights()

    def _init_weights(self):
        r"""Initialize to near-zero output at start."""
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        # 其他层用 xavier
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, lut_out, sdr):
        r"""Forward function.
        
        Args:
            lut_out (Tensor): LUT output image, shape (B, 3, H, W).
            sdr (Tensor): Original SDR input, shape (B, 3, H, W).
        
        Returns:
            Tensor: Refined HDR image.
            Tensor: Residual for debugging/loss.
        """
        H, W = lut_out.shape[2:]
        
        # 构建输入特征
        x = torch.cat([lut_out, sdr], dim=1)  # (B, 6, H, W)
        
        if self.scale_factor > 1:
            # 下采样处理
            H_low = H // self.scale_factor
            W_low = W // self.scale_factor
            x = F.interpolate(x, size=(H_low, W_low), mode='bilinear', align_corners=True)
        
        # 多尺度特征提取
        f1 = self.act(self.conv1(x))
        f2 = self.act(self.conv2(f1))
        f3 = self.act(self.conv3(f2))
        f4 = self.act(self.conv4(f3 + f1))  # skip connection
        
        # 输出残差
        res = torch.tanh(self.conv_out(f4)) * self.residual_scale
        
        if self.scale_factor > 1:
            # 上采样回原分辨率
            res = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=True)
        
        refined = lut_out + res
        
        return refined, res


@MODELS.register_module()
class AiLUT(BaseModel):
    r"""Adaptive-Interval 3D Lookup Table for real-time image enhancement.

    Args:
        n_ranks (int, optional): Number of ranks in the mapping h
            (or the number of basis LUTs). Default: 3.
        n_vertices (int, optional): Number of sampling points along
            each lattice dimension. Default: 33.
        en_adaint (bool, optional): Whether to enable AdaInt. Default: True.
        en_adaint_share (bool, optional): Whether to enable Share-AdaInt.
            Only used when `en_adaint` is True. Default: False.
        backbone (str, optional): Backbone architecture to use. Can be either 'tpami'
            or 'res18'. Default: 'tpami'.
        pretrained (bool, optional): Whether to use ImageNet-pretrained weights.
            Only used when `backbone` is 'res18'. Default: None.
        n_colors (int, optional): Number of input color channels. Default: 3.
        sparse_factor (float, optional): Loss weight for the sparse regularization term.
            Default: 0.0001.
        smooth_factor (float, optional): Loss weight for the smoothness regularization term.
            Default: 0.
        monotonicity_factor (float, optional): Loss weight for the monotonicaity
            regularization term. Default: 10.0.
        en_spatial_offset (bool, optional): Whether to enable spatial-adaptive offset
            for LUT lookup. This breaks the "same color same output" limitation. Default: False.
        offset_hidden (int, optional): Hidden dimension for offset generator. Default: 16.
        offset_scale_factor (int, optional): Downsampling factor for offset generator. Default: 4.
        offset_scale (float, optional): Maximum offset magnitude. Default: 0.05.
        recons_loss (dict, optional): Config for pixel-wise reconstruction loss.
        train_cfg (dict, optional): Config for training. Default: None.
        test_cfg (dict, optional): Config for testing. Default: None.
    """

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
        n_ranks=3,
        n_vertices=33,
        en_adaint=True,
        en_adaint_share=False,
        backbone='tpami',
        pretrained=False,
        n_colors=3,
        sparse_factor=0.0001,
        smooth_factor=0,
        monotonicity_factor=10.0,
        en_refine=False,
        en_pre_refine=False,
        en_post_refine=False,
        en_spatial_offset=False,
        en_input_smooth=True,
        en_deblocking=False,
        refine_hidden=16,
        refine_scale_factor=4,
        pre_refine_residual_scale=0.1,
        post_refine_residual_scale=0.1,
        deblocking_hidden=24,
        deblocking_scale_factor=2,
        deblocking_residual_scale=0.15,
        offset_hidden=16,
        offset_scale_factor=4,
        offset_scale=0.05,
        offset_mode='output',
        smooth_scale_factor=4,
        smooth_kernel_size=5,
        residual_reg_factor=0.0,
        recons_loss=dict(type='L2Loss', loss_weight=1.0, reduction='mean'),
        train_cfg=None,
        test_cfg=None):

        super().__init__()

        assert backbone.lower() in ['tpami', 'res18']

        # mapping f
        self.backbone = dict(
            tpami=TPAMIBackbone,
            res18=Res18Backbone)[backbone.lower()](pretrained, extra_pooling=en_adaint)

        # mapping h
        self.lut_generator = LUTGenerator(
            n_colors, n_vertices, self.backbone.out_channels, n_ranks)

        # mapping g
        if en_adaint:
            self.adaint = AdaInt(
                n_colors, n_vertices, self.backbone.out_channels, en_adaint_share)
        else:
            uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1) \
                                    .repeat(n_colors, 1)
            self.register_buffer('uniform_vertices', uniform_vertices.unsqueeze(0))

        # Spatial-Adaptive LUT module (new: 同时处理两种块状伪影)
        self.en_spatial_offset = en_spatial_offset
        self.en_input_smooth = en_input_smooth
        self.smooth_scale_factor = smooth_scale_factor
        self.smooth_kernel_size = smooth_kernel_size
        self.offset_mode = offset_mode
        if en_spatial_offset:
            self.spatial_adaptive = SpatialAdaptiveLUT(
                hidden=offset_hidden,
                offset_scale_factor=offset_scale_factor,
                offset_scale=offset_scale,
                smooth_scale_factor=smooth_scale_factor,
                smooth_kernel_size=smooth_kernel_size,
                en_smooth=en_input_smooth,
                offset_mode=offset_mode
            )

        # Pre-refinement head (before LUT, optional) - 保留兼容性
        self.en_pre_refine = en_pre_refine
        if en_pre_refine:
            self.pre_refine = PreRefineHead(
                in_channels=3,
                hidden=refine_hidden,
                scale_factor=refine_scale_factor,
                residual_scale=pre_refine_residual_scale
            )

        # Post-refinement head (after LUT, optional)
        self.en_post_refine = en_post_refine
        self.en_deblocking = en_deblocking
        if en_post_refine:
            self.post_refine = PostRefineHead(
                in_channels=6,
                hidden=refine_hidden,
                scale_factor=refine_scale_factor,
                residual_scale=post_refine_residual_scale
            )
        
        # Lightweight Deblocking module (recommended for artifact removal)
        if en_deblocking:
            self.deblocking = LightweightDeblocking(
                hidden=deblocking_hidden,
                scale_factor=deblocking_scale_factor,
                residual_scale=deblocking_residual_scale
            )
        
        # 兼容旧配置
        self.en_refine = en_refine or en_pre_refine or en_post_refine

        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.en_adaint = en_adaint
        self.sparse_factor = sparse_factor
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor
        self.residual_reg_factor = residual_reg_factor
        self.offset_scale = offset_scale
        self.backbone_name = backbone.lower()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False

        self.init_weights()

        self.recons_loss = build_loss(recons_loss)

        # fix AdaInt for some steps
        self.n_fix_iters = train_cfg.get('n_fix_iters', 0) if train_cfg else 0
        # fix Refinement for some steps (new)
        self.n_fix_refine_iters = train_cfg.get('n_fix_refine_iters', 0) if train_cfg else 0
        # freeze LUT modules for stage2 training (new)
        self.freeze_lut = train_cfg.get('freeze_lut', False) if train_cfg else False
        self.adaint_fixed = False
        self.refine_fixed = False
        self.register_buffer('cnt_iters', torch.zeros(1))
        
        # 如果是阶段2训练，冻结 LUT 相关模块
        if self.freeze_lut:
            self._freeze_lut_modules()

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
        For the mapping g (`adaint`), we use all-zero and all-one initializations for its weights
        and bias, respectively.
        """
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        if self.backbone_name not in ['res18']:
            self.apply(special_initilization)
        self.lut_generator.init_weights()
        if self.en_adaint:
            self.adaint.init_weights()

    def _freeze_lut_modules(self):
        r"""Freeze LUT-related modules for stage2 training."""
        # 冻结 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 冻结 lut_generator
        for param in self.lut_generator.parameters():
            param.requires_grad = False
        # 冻结 adaint
        if self.en_adaint:
            for param in self.adaint.parameters():
                param.requires_grad = False
        get_root_logger().info('Frozen LUT modules (backbone, lut_generator, adaint) for stage2 training.')

    def forward_dummy(self, imgs):
        r"""The real implementation of model forward.

        Args:
            img (Tensor): Input image, shape (b, c, h, w).
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
                Output image, LUT weights, Sampling Coordinates, 
                Pre-refine residual (if enabled), Post-refine residual (if enabled),
                Spatial offset (if enabled), Deblocking residual (if enabled).
        """
        # E: (b, f) - Backbone 用原始图像提取特征
        codes = self.backbone(imgs)
        # (b, m), T: (b, c, d, d, d)
        weights, luts = self.lut_generator(codes)
        # \hat{P}: (b, c, d)
        if self.en_adaint:
            vertices = self.adaint(codes)
        else:
            vertices = self.uniform_vertices

        # Pre-refinement: 平滑输入图像 (可选，保留兼容性)
        pre_refine_residual = None
        if self.en_pre_refine:
            imgs_for_lut, pre_refine_residual = self.pre_refine(imgs)
        else:
            imgs_for_lut = imgs

        # Spatial offset 处理
        spatial_offset = None
        if self.en_spatial_offset:
            if self.offset_mode == 'input':
                # 作用于输入：修改查表坐标
                imgs_for_lut, spatial_offset, _ = self.spatial_adaptive(imgs_for_lut)
            else:
                # 作用于输出：先获取 offset，后面加到 LUT 输出上
                spatial_offset = self.spatial_adaptive(imgs)

        # LUT 变换
        lut_out = ailut_transform(imgs_for_lut, luts, vertices)

        # 如果 offset 作用于输出，在这里加上
        if self.en_spatial_offset and self.offset_mode == 'output':
            lut_out = lut_out + spatial_offset

        # Post-refinement: 修复 LUT 输出的伪影 (可选，旧版)
        post_refine_residual = None
        if self.en_post_refine:
            outs, post_refine_residual = self.post_refine(lut_out, imgs)
        else:
            outs = lut_out

        # Lightweight Deblocking: 专门针对块状伪影的后处理 (推荐)
        deblocking_residual = None
        if self.en_deblocking:
            outs, deblocking_residual = self.deblocking(outs, imgs)

        return outs, weights, vertices, pre_refine_residual, post_refine_residual, spatial_offset, deblocking_residual

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        r"""Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor, optional): Ground-truth image. Default: None.
            test_mode (bool, optional): Whether in test mode or not. Default: False.
            kwargs (dict, optional): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        r"""Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            outputs (dict): Output results.
        """
        losses = dict()
        output, weights, vertices, pre_refine_residual, post_refine_residual, spatial_offset, deblocking_residual = self.forward_dummy(lq)

        # Reconstruction loss
        losses['loss_recons'] = self.recons_loss(output, gt)

        # Sparse regularization for LUT weights
        if self.sparse_factor > 0:
            losses['loss_sparse'] = self.sparse_factor * torch.mean(weights.pow(2))

        # LUT smoothness and monotonicity regularization
        reg_smoothness, reg_monotonicity = self.lut_generator.regularizations(
            self.smooth_factor, self.monotonicity_factor)
        if self.smooth_factor > 0:
            losses['loss_smooth'] = reg_smoothness
        if self.monotonicity_factor > 0:
            losses['loss_mono'] = reg_monotonicity

        # Refinement losses
        iter_count = int(self.cnt_iters.item())
        
        # Pre-refine debug
        if self.en_pre_refine and pre_refine_residual is not None:
            if iter_count % 100 == 0:
                print(f"[DEBUG iter={iter_count}] pre_refine: min={pre_refine_residual.min().item():.6f}, max={pre_refine_residual.max().item():.6f}, abs_mean={pre_refine_residual.abs().mean().item():.6f}")
            if self.residual_reg_factor > 0:
                losses['loss_pre_res_reg'] = self.residual_reg_factor * pre_refine_residual.abs().mean()

        # Post-refine debug
        if self.en_post_refine and post_refine_residual is not None:
            if iter_count % 100 == 0:
                print(f"[DEBUG iter={iter_count}] post_refine: min={post_refine_residual.min().item():.6f}, max={post_refine_residual.max().item():.6f}, abs_mean={post_refine_residual.abs().mean().item():.6f}")
            if self.residual_reg_factor > 0:
                losses['loss_post_res_reg'] = self.residual_reg_factor * post_refine_residual.abs().mean()

        # Spatial offset debug
        if self.en_spatial_offset and spatial_offset is not None:
            if iter_count % 100 == 0:
                print(f"[DEBUG iter={iter_count}] spatial_offset: min={spatial_offset.min().item():.6f}, max={spatial_offset.max().item():.6f}, abs_mean={spatial_offset.abs().mean().item():.6f}")

        # Deblocking debug
        if self.en_deblocking and deblocking_residual is not None:
            if iter_count % 100 == 0:
                print(f"[DEBUG iter={iter_count}] deblocking: min={deblocking_residual.min().item():.6f}, max={deblocking_residual.max().item():.6f}, abs_mean={deblocking_residual.abs().mean().item():.6f}")

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        r"""Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor, optional): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool, optional): Whether to save image. Default: False.
            save_path (str, optional): Path to save image. Default: None.
            iteration (int, optional): Iteration for the saving image name.
                Default: None.
        Returns:
            outputs (dict): Output results.
        """
        output, _, _, _, _, _, _ = self.forward_dummy(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            output_img = tensor2img(output, out_type=np.uint16)
            output_img = output_img[:, :, ::-1]  # BGR -> RGB，与 test.py 保持一致
            mmcv.imwrite(output_img, save_path)

        return results

    def train_step(self, data_batch, optimizer):
        r"""Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.
        Returns:
            dict: Returned output.
        """
        # fix AdaInt in the first several epochs
        if self.en_adaint and self.cnt_iters < self.n_fix_iters:
            if not self.adaint_fixed:
                self.adaint_fixed = True
                self.adaint.requires_grad_(False)
                get_root_logger().info(f'Fix AdaInt for {self.n_fix_iters} iters.')
        elif self.en_adaint and self.cnt_iters == self.n_fix_iters:
            self.adaint.requires_grad_(True)
            if self.adaint_fixed:
                self.adaint_fixed = False
                get_root_logger().info(f'Unfix AdaInt after {self.n_fix_iters} iters.')

        # fix Refinement heads in the first several epochs (let LUT learn first)
        if (self.en_pre_refine or self.en_post_refine) and self.cnt_iters < self.n_fix_refine_iters:
            if not self.refine_fixed:
                self.refine_fixed = True
                if self.en_pre_refine:
                    self.pre_refine.requires_grad_(False)
                if self.en_post_refine:
                    self.post_refine.requires_grad_(False)
                get_root_logger().info(f'Fix Refine heads for {self.n_fix_refine_iters} iters.')
        elif (self.en_pre_refine or self.en_post_refine) and self.cnt_iters == self.n_fix_refine_iters:
            if self.en_pre_refine:
                self.pre_refine.requires_grad_(True)
            if self.en_post_refine:
                self.post_refine.requires_grad_(True)
            if self.refine_fixed:
                self.refine_fixed = False
                get_root_logger().info(f'Unfix Refine heads after {self.n_fix_refine_iters} iters.')

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.update({'log_vars': log_vars})

        self.cnt_iters += 1
        return outputs

    def val_step(self, data_batch, **kwargs):
        r"""Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict, optional): Other arguments for ``val_step``.
        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def evaluate(self, output, gt):
        r"""Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            dict: Evaluation results.
        """
        import cv2
        crop_border = self.test_cfg.crop_border

        # 直接用 tensor 计算，保持 float32 精度，不量化到 8-bit
        output_np = output.squeeze(0).float().detach().cpu().clamp_(0, 1).numpy()
        gt_np = gt.squeeze(0).float().detach().cpu().clamp_(0, 1).numpy()
        
        # CHW -> HWC
        output_np = output_np.transpose(1, 2, 0)
        gt_np = gt_np.transpose(1, 2, 0)
        
        if crop_border != 0:
            output_np = output_np[crop_border:-crop_border, crop_border:-crop_border, :]
            gt_np = gt_np[crop_border:-crop_border, crop_border:-crop_border, :]

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if metric == 'PSNR':
                mse = np.mean((output_np - gt_np) ** 2)
                if mse == 0:
                    eval_result['PSNR'] = float('inf')
                else:
                    eval_result['PSNR'] = 20. * np.log10(1.0 / np.sqrt(mse))  # max=1.0 for [0,1] range
            elif metric == 'SSIM':
                # Float32 SSIM，范围 [0,1]
                C1 = (0.01) ** 2  # 对应 max=1.0
                C2 = (0.03) ** 2
                ssims = []
                for i in range(output_np.shape[2]):
                    img1 = output_np[..., i].astype(np.float64)
                    img2 = gt_np[..., i].astype(np.float64)
                    kernel = cv2.getGaussianKernel(11, 1.5)
                    window = np.outer(kernel, kernel.transpose())
                    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
                    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
                    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
                    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
                    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
                    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
                    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                    ssims.append(ssim_map.mean())
                eval_result['SSIM'] = np.mean(ssims)
        return eval_result
