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
# Import losses to ensure they are registered before build_loss is called
from mmedit.models import losses  # noqa: F401
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

    def regularizations(self, smoothness, monotonicity, curvature_weight=0.0, mono_delta=0.0):
        """LUT regularization with curvature smoothing and soft-monotonicity.

        Args:
            smoothness: Weight for TV smoothness (original, can be 0)
            monotonicity: Weight for monotonicity constraint
            curvature_weight: Weight for 2nd-order curvature on RGB axes (new)
            mono_delta: Tolerance for soft-monotonicity, allows small local rollback (new)
        """
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn, curv = 0, 0, 0

        # BT.709 luminance coefficients for R, G, B axes
        # Weight curvature by each axis's contribution to luminance
        lum_weights = [0.2126, 0.7152, 0.0722]  # R, G, B

        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            # Soft-Mono: allow small local rollback for smoother transitions
            mn += F.relu(diff - mono_delta).sum(0).mean()

            # 2nd-order curvature: all RGB axes, weighted by luminance contribution
            if curvature_weight > 0:
                d2 = torch.diff(diff, dim=i)
                # Highlight region: last 30% of bins
                high_start = int(0.7 * (self.n_vertices - 2))
                # Weight by BT.709 luminance coefficient (i=2→R, i=3→G, i=4→B)
                axis_weight = lum_weights[i - 2]
                # Use L2 (sqrt of squared) instead of L1 - more sensitive to banding
                curv += axis_weight * torch.sqrt(d2[..., high_start:].pow(2) + 1e-6).mean()

        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        reg_curvature = curvature_weight * curv
        return reg_smoothness, reg_monotonicity, reg_curvature


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


@MODELS.register_module(force=True)
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
        mono_delta (float, optional): Tolerance for soft-monotonicity constraint.
            Allows small local rollback for smoother color transitions. Default: 0.0.
        curvature_factor (float, optional): Loss weight for 2nd-order curvature smoothing
            on luminance axis. Reduces highlight banding. Default: 0.0.
        chroma_smooth_weight (float, optional): Weight for chroma smoothness loss in
            highlight regions. Reduces color edge artifacts. Default: 0.0.
        highlight_charb_weight (float, optional): Weight for using Charbonnier loss
            in highlight regions instead of MSE. Default: 0.0.
        highlight_sampling_gamma (float, optional): Gamma value for warping AdaInt vertices
            to densify highlight sampling. gamma > 1 densifies highlights (recommended: 2.2),
            gamma = 1 no change, gamma < 1 densifies shadows. Default: 1.0.
        highlight_gradient_weight (float, optional): Weight for gradient smoothness loss
            applied ONLY in highlight regions. Reduces blocking artifacts. Default: 0.0.
        highlight_chroma_weight (float, optional): Weight for chroma smoothness loss
            applied ONLY in highlight regions. Uses proper chroma=RGB/luma. Default: 0.0.
        edge_aware_weight (float, optional): Weight for edge-aware reconstruction loss.
            When > 0, edges detected using Sobel operators will be weighted more heavily
            in the reconstruction loss. Default: 0.0.
        recons_loss (dict, optional): Config for pixel-wise reconstruction loss.
        perceptual_loss (dict, optional): Config for perceptual loss. If None, perceptual
            loss will not be used. Default: None.
        gradient_loss (dict, optional): Config for gradient loss. If None, gradient
            loss will not be used. Default: None.
        gamut_loss (dict, optional): Config for color gamut loss. If None, gamut
            loss will not be used. Default: None.
        hdr_tone_loss (dict, optional): Config for HDR tone mapping loss. If None, HDR
            tone loss will not be used. Default: None.
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
        mono_delta=0.0,
        curvature_factor=0.0,
        chroma_smooth_weight=0.0,
        highlight_charb_weight=0.0,
        highlight_sampling_gamma=1.0,
        highlight_gradient_weight=0.0,
        highlight_chroma_weight=0.0,
        edge_aware_weight=0.0,
        recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
        perceptual_loss=None,
        gradient_loss=None,
        gamut_loss=None,
        hdr_tone_loss=None,
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

        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.en_adaint = en_adaint
        self.sparse_factor = sparse_factor
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor
        self.mono_delta = mono_delta
        self.curvature_factor = curvature_factor
        self.chroma_smooth_weight = chroma_smooth_weight
        self.highlight_charb_weight = highlight_charb_weight
        self.highlight_sampling_gamma = highlight_sampling_gamma
        self.highlight_gradient_weight = highlight_gradient_weight
        self.highlight_chroma_weight = highlight_chroma_weight
        self.edge_aware_weight = edge_aware_weight
        self.backbone_name = backbone.lower()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False

        self.init_weights()

        self.recons_loss = build_loss(recons_loss)

        # Build new loss modules
        self.perceptual_loss = build_loss(perceptual_loss) if perceptual_loss else None
        self.gradient_loss = build_loss(gradient_loss) if gradient_loss else None
        self.gamut_loss = build_loss(gamut_loss) if gamut_loss else None
        self.hdr_tone_loss = build_loss(hdr_tone_loss) if hdr_tone_loss else None

        # fix AdaInt for some steps
        self.n_fix_iters = train_cfg.get('n_fix_iters', 0) if train_cfg else 0
        self.adaint_fixed = False
        self.register_buffer('cnt_iters', torch.zeros(1))

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

    def forward_dummy(self, imgs):
        r"""The real implementation of model forward.

        Args:
            img (Tensor): Input image, shape (b, c, h, w).
        Returns:
            tuple(Tensor, Tensor, Tensor):
                Output image, LUT weights, Sampling Coordinates.
        """
        # E: (b, f)
        codes = self.backbone(imgs)
        # (b, m), T: (b, c, d, d, d)
        weights, luts = self.lut_generator(codes)
        # \hat{P}: (b, c, d)
        if self.en_adaint:
            vertices = self.adaint(codes)

            # Gamma warp to densify ONLY highlight sampling (index-based)
            # Only affects vertices after pivot_idx, leaves dark/mid regions untouched
            # gamma > 1: densifies highlights (recommended: 2.2)
            # gamma = 1: no change
            if self.highlight_sampling_gamma != 1.0:
                gamma = self.highlight_sampling_gamma
                pivot = 0.7
                n = vertices.shape[-1]
                pivot_idx = int(pivot * (n - 1))

                out = vertices.clone()
                # Use actual vertex value at pivot_idx as reference (more accurate than fixed 0.7)
                v_pivot = vertices[..., pivot_idx:pivot_idx+1]
                # Normalize highlight region to [0, 1], apply gamma, remap back
                denom = (1.0 - v_pivot).clamp(min=1e-6)
                vh = (vertices[..., pivot_idx:] - v_pivot) / denom
                # Clamp vh to [0, 1] to avoid NaN from negative values raised to non-integer power
                vh = vh.clamp(min=0.0, max=1.0)
                out[..., pivot_idx:] = v_pivot + (vh ** gamma) * (1.0 - v_pivot)

                vertices = out
                # Renormalize to ensure last vertex is 1.0
                vertices = vertices / vertices[:, :, -1:].clamp(min=1e-6)
        else:
            vertices = self.uniform_vertices

        outs = ailut_transform(imgs, luts, vertices)

        return outs, weights, vertices

    def _compute_edge_weight(self, img):
        r"""Compute edge weight map using Sobel operators.

        Args:
            img (Tensor): Input image, shape (b, c, h, w).

        Returns:
            Tensor: Edge weight map, shape (b, 1, h, w).
        """
        # Sobel kernels
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3).to(img)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3).to(img)

        # Convert to grayscale for edge detection
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        # Compute gradients
        grad_x = F.conv2d(gray, kx, padding=1)
        grad_y = F.conv2d(gray, ky, padding=1)
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize and create weight map
        edge_weight = 1.0 + self.edge_aware_weight * edge_magnitude
        edge_weight = edge_weight / edge_weight.mean()  # Normalize to maintain scale

        return edge_weight

    def _rgb_to_luma(self, x):
        """Convert RGB to luminance using BT.709 coefficients."""
        w = x.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)

    def _preprocess_deblock(self, x, luma_th=0.80, grad_th=0.03):
        """Deblock for highlight flat regions to reduce 16x16 JPEG block artifacts.
        
        Uses multi-scale approach: detects 16x16 block boundaries and smooths them.
        
        Args:
            x (Tensor): Input image, shape (b, c, h, w) in [0, 1].
            luma_th (float): Luminance threshold for highlight detection.
            grad_th (float): Gradient threshold for flat region detection.
            
        Returns:
            Tensor: Preprocessed image with reduced block artifacts.
        """
        b, c, h, w = x.shape
        
        # Step 1: Compute luminance
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        
        # Step 2: Compute gradient magnitude
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        grad_x = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode='reflect'), sobel_x)
        grad_y = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode='reflect'), sobel_y)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Step 3: Soft mask for highlight flat regions
        mask_luma = torch.sigmoid((luma - luma_th) / 0.05)
        mask_grad = torch.sigmoid((grad_th - grad) / 0.008)
        mask = mask_luma * mask_grad
        
        # Step 4: Multi-pass blur to handle 16x16 blocks
        # First pass: 5x5 Gaussian
        kernel_5 = torch.tensor([[1, 4, 6, 4, 1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4, 6, 4, 1]], 
                                dtype=x.dtype, device=x.device) / 256.0
        kernel_5 = kernel_5.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        x_blur = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), kernel_5, groups=3)
        
        # Second pass: another 5x5 for stronger smoothing in masked areas
        x_blur2 = F.conv2d(F.pad(x_blur, (2, 2, 2, 2), mode='reflect'), kernel_5, groups=3)
        
        # Adaptive blend: stronger blur where mask is higher
        x_smooth = x_blur * (1 - mask) + x_blur2 * mask
        
        # Step 5: Blend with original based on mask
        x = x * (1 - mask) + x_smooth * mask
        
        # Step 6: Stronger dither to break block synchronization
        noise = (torch.rand_like(x) - 0.5) * (1.0 / 255.0)
        x = torch.clamp(x + mask * noise, 0.0, 1.0)
        
        return x

    def _compute_highlight_mask(self, img, soft=True):
        """Compute dual-zone highlight mask based on luminance.
        
        Uses two overlapping sigmoid regions:
        - Shoulder region (0.45+): catches mid-to-high transition where blocking starts
        - Extreme highlight (0.75+): stronger weight for very bright areas

        Args:
            img (Tensor): Input image, shape (b, c, h, w).
            soft (bool): If True, use soft sigmoid mask. If False, use hard threshold.

        Returns:
            Tensor: Highlight mask, shape (b, 1, h, w), values in [0, 1].
        """
        luma = self._rgb_to_luma(img)
        if soft:
            # Dual-zone soft mask for better coverage
            # Shoulder region: catches blocking that starts in mid-highlights
            mask_shoulder = torch.sigmoid((luma - 0.45) / 0.15)
            # Extreme highlight: stronger weight for very bright areas
            mask_extreme = torch.sigmoid((luma - 0.75) / 0.08)
            # Combine both zones
            mask = torch.clamp(mask_shoulder + mask_extreme, 0, 1)
        else:
            # Hard mask (original behavior)
            mask = torch.clamp((luma - 0.7) / 0.3, 0, 1)
        return mask

    def _highlight_aware_recons_loss(self, pred, target):
        """Compute reconstruction loss with Charbonnier in highlight regions.

        Uses MSE for mid/low tones and Charbonnier for highlights to reduce blocking.
        
        NOTE: mask is computed from PRED, not GT. This allows the network to first
        increase brightness, then apply smoothing once pred enters highlight range.
        Using GT mask would prematurely smooth mid-tones that should become highlights.

        Args:
            pred (Tensor): Predicted image, shape (b, c, h, w).
            target (Tensor): Ground truth image, shape (b, c, h, w).

        Returns:
            Tensor: Mixed reconstruction loss.
        """
        # Use pred for mask: smooth only when pred is already in highlight range
        highlight_mask = self._compute_highlight_mask(pred)

        diff_sq = (pred - target) ** 2
        loss_mse = diff_sq
        loss_charb = torch.sqrt(diff_sq + 1e-6)

        # Mix: MSE for non-highlight, Charbonnier for highlight
        mixed_loss = (1 - highlight_mask) * loss_mse + highlight_mask * loss_charb
        return mixed_loss.mean()

    def _highlight_gradient_loss(self, pred):
        """Compute gradient smoothness loss ONLY in highlight regions.
        
        Penalizes sharp gradients in highlights to reduce blocking artifacts.
        Weight = mask * luma to focus on bright areas where blocking actually occurs.
        
        Args:
            pred (Tensor): Predicted image, shape (b, c, h, w).
            
        Returns:
            Tensor: Highlight gradient loss.
        """
        luma = self._rgb_to_luma(pred)
        mask = self._compute_highlight_mask(pred, soft=True)
        
        # Weight = mask * luma: focus loss on bright areas where blocking occurs
        weight = mask * luma.clamp(min=0.3)
        
        # Spatial gradients
        gx = pred[..., :, 1:] - pred[..., :, :-1]
        gy = pred[..., 1:, :] - pred[..., :-1, :]
        
        # Align weight with gradient dimensions
        weight_x = weight[..., :, 1:]
        weight_y = weight[..., 1:, :]
        
        return (gx.abs() * weight_x).mean() + (gy.abs() * weight_y).mean()

    def _highlight_chroma_loss(self, pred):
        """Compute chroma smoothness loss ONLY in highlight regions.
        
        Uses LOG-CHROMA instead of RGB/luma for HDR stability.
        In extreme highlights, RGB/luma ≈ [1,1,1] has no gradient.
        Log-chroma = log(RGB) - log(luma) preserves sensitivity in HDR range.
        
        Args:
            pred (Tensor): Predicted image, shape (b, c, h, w).
            
        Returns:
            Tensor: Highlight chroma loss.
        """
        luma = self._rgb_to_luma(pred)
        mask = self._compute_highlight_mask(pred, soft=True)
        
        # Weight = mask * luma: focus on bright areas
        weight = mask * luma.clamp(min=0.3)
        
        # Log-chroma: stable in HDR extreme highlights
        # Clamp to positive values before log to avoid NaN
        pred_safe = pred.clamp(min=1e-6)
        luma_safe = luma.clamp(min=1e-6)
        log_rgb = torch.log(pred_safe + 1e-3)
        log_luma = torch.log(luma_safe + 1e-3)
        chroma = log_rgb - log_luma
        
        # Spatial gradients of log-chroma
        gx = chroma[..., :, 1:] - chroma[..., :, :-1]
        gy = chroma[..., 1:, :] - chroma[..., :-1, :]
        
        # Align weight with gradient dimensions
        weight_x = weight[..., :, 1:]
        weight_y = weight[..., 1:, :]
        
        return (gx.abs() * weight_x).mean() + (gy.abs() * weight_y).mean()

    def _chroma_smooth_loss(self, pred, target):
        """Compute chroma smoothness loss in highlight regions (legacy).

        Reduces color edge artifacts by encouraging smooth chroma transitions.
        
        NOTE: mask is computed from PRED to allow brightness increase first.

        Args:
            pred (Tensor): Predicted image, shape (b, c, h, w).
            target (Tensor): Ground truth image, shape (b, c, h, w).

        Returns:
            Tensor: Chroma smoothness loss.
        """
        # Compute luminance
        Y_pred = self._rgb_to_luma(pred)

        # Chroma = RGB / luma (proper chroma definition)
        chroma = pred / Y_pred.clamp(min=1e-6)

        # Highlight mask from PRED (not GT) - smooth only when pred is in highlight
        mask = self._compute_highlight_mask(pred, soft=True)

        # Spatial smoothness on chroma
        dx = torch.abs(chroma[:, :, :, :-1] - chroma[:, :, :, 1:])
        dy = torch.abs(chroma[:, :, :-1, :] - chroma[:, :, 1:, :])

        loss = (dx * mask[:, :, :, :-1]).mean() + (dy * mask[:, :, :-1, :]).mean()
        return loss

    def _weighted_recons_loss(self, pred, target, weight):
        r"""Compute weighted reconstruction loss.

        Args:
            pred (Tensor): Predicted image, shape (b, c, h, w).
            target (Tensor): Ground truth image, shape (b, c, h, w).
            weight (Tensor): Weight map, shape (b, 1, h, w).

        Returns:
            Tensor: Weighted reconstruction loss.
        """
        loss = (pred - target) ** 2
        weighted_loss = loss * weight
        return weighted_loss.mean()

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
        output, weights, vertices = self.forward_dummy(lq)

        # Reconstruction loss
        if self.highlight_charb_weight > 0:
            # Highlight-aware: MSE for mid/low, Charbonnier for highlights
            losses['loss_recons'] = self.highlight_charb_weight * self._highlight_aware_recons_loss(output, gt)
        elif self.edge_aware_weight > 0:
            # Edge-aware weighting
            edge_weight = self._compute_edge_weight(gt)
            losses['loss_recons'] = self._weighted_recons_loss(output, gt, edge_weight)
        else:
            # Standard reconstruction loss
            losses['loss_recons'] = self.recons_loss(output, gt)

        # Chroma smoothness loss for highlight regions (legacy)
        if self.chroma_smooth_weight > 0:
            losses['loss_chroma_smooth'] = self.chroma_smooth_weight * self._chroma_smooth_loss(output, gt)

        # NEW: Highlight-only gradient loss (replaces global gradient loss for HDR)
        if self.highlight_gradient_weight > 0:
            losses['loss_highlight_grad'] = self.highlight_gradient_weight * self._highlight_gradient_loss(output)

        # NEW: Highlight-only chroma loss (proper chroma = RGB/luma)
        if self.highlight_chroma_weight > 0:
            losses['loss_highlight_chroma'] = self.highlight_chroma_weight * self._highlight_chroma_loss(output)

        # Perceptual loss
        if self.perceptual_loss is not None:
            loss_percep, loss_style = self.perceptual_loss(output, gt)
            if loss_percep is not None:
                losses['loss_perceptual'] = loss_percep
            if loss_style is not None:
                losses['loss_style'] = loss_style

        # Gradient loss
        if self.gradient_loss is not None:
            losses['loss_gradient'] = self.gradient_loss(output, gt)

        # Color gamut loss
        if self.gamut_loss is not None:
            losses['loss_gamut'] = self.gamut_loss(output, gt)

        # HDR tone mapping loss
        if self.hdr_tone_loss is not None:
            losses['loss_hdr_tone'] = self.hdr_tone_loss(output, gt)

        # Existing regularization losses
        if self.sparse_factor > 0:
            losses['loss_sparse'] = self.sparse_factor * torch.mean(weights.pow(2))

        # LUT regularization with new curvature and soft-mono
        reg_smoothness, reg_monotonicity, reg_curvature = self.lut_generator.regularizations(
            self.smooth_factor, self.monotonicity_factor,
            self.curvature_factor, self.mono_delta)
        if self.smooth_factor > 0:
            losses['loss_smooth'] = reg_smoothness
        if self.monotonicity_factor > 0:
            losses['loss_mono'] = reg_monotonicity
        if self.curvature_factor > 0:
            losses['loss_curvature'] = reg_curvature

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
        # Deblock + dither for highlight flat regions to reduce JPEG block artifacts
        #lq = self._preprocess_deblock(lq)

        output, _, _ = self.forward_dummy(lq)
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
            mmcv.imwrite(tensor2img(output,  out_type=np.uint16), save_path)

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
        crop_border = self.test_cfg.crop_border

        # Use uint16 for HDR data to preserve more precision (0-65535 range)
        output = tensor2img(output, out_type=np.uint16, min_max=(0, 1))
        gt = tensor2img(gt, out_type=np.uint16, min_max=(0, 1))

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](
                output, gt, crop_border)
        return eval_result
