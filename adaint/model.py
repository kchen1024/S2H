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
            mode='bilinear', align_corners=False)
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


class SpatialRefineHead(nn.Module):
    r"""Lightweight Spatial Refinement Head for LUT output.

    Learns residual corrections to compensate for LUT's discrete sampling artifacts,
    especially in highlight regions with halo/banding issues.

    Args:
        in_channels (int): Number of input channels (lut_out + lq). Default: 6.
        hidden (int): Hidden layer dimension. Default: 24.
        use_backbone_feat (bool): Whether to use backbone features for global modulation. Default: False.
        backbone_feat_dim (int): Dimension of backbone features. Default: 512.
    """

    def __init__(self, in_channels=6, hidden=24, use_backbone_feat=False, backbone_feat_dim=512):
        super().__init__()
        self.use_backbone_feat = use_backbone_feat

        # Global feature projection (optional)
        if use_backbone_feat:
            self.feat_proj = nn.Sequential(
                nn.Linear(backbone_feat_dim, hidden),
                nn.ReLU(inplace=True)
            )

        # Local feature extraction - use LeakyReLU to avoid dead neurons
        self.local_feat = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Residual prediction - direct 3 channel output
        self.residual_head = nn.Conv2d(hidden, 3, 1)

        # Adaptive mask (learns where to refine) - for visualization
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        r"""Initialize weights for near-identity mapping at start."""
        # Use default kaiming init for conv layers (already done by PyTorch)
        # Just ensure residual_head starts small
        nn.init.xavier_uniform_(self.residual_head.weight, gain=0.5)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, lut_out, lq, backbone_feat=None):
        r"""Forward function.

        Args:
            lut_out (Tensor): LUT transformed output, shape (B, 3, H, W).
            lq (Tensor): Original input image, shape (B, 3, H, W).
            backbone_feat (Tensor, optional): Backbone feature vector, shape (B, feat_dim).

        Returns:
            tuple(Tensor, Tensor):
                - refined: Refined output, shape (B, 3, H, W).
                - mask: Refinement mask, shape (B, 1, H, W).
        """
        x = torch.cat([lut_out, lq], dim=1)  # (B, 6, H, W)

        feat = self.local_feat(x)  # (B, hidden, H, W)

        # Global modulation with backbone features (optional)
        if self.use_backbone_feat and backbone_feat is not None:
            global_feat = self.feat_proj(backbone_feat)  # (B, hidden)
            global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)  # (B, hidden, 1, 1)
            feat = feat * (1 + global_feat)

        # Predict residual - scale to small range [-0.05, 0.05]
        residual = torch.tanh(self.residual_head(feat)) * 0.05
        
        # Mask for visualization/analysis only
        mask = self.mask_head(feat)

        # Direct residual addition
        refined = lut_out + residual

        return refined, mask, residual  # 返回 residual 用于调试


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
        en_refine (bool, optional): Whether to enable spatial refinement head. Default: False.
        refine_hidden (int, optional): Hidden dimension for refinement head. Default: 24.
        refine_use_backbone_feat (bool, optional): Whether to use backbone features
            for global modulation in refinement. Default: False.
        mask_sparse_factor (float, optional): Loss weight for mask sparsity regularization.
            Prevents refinement from taking over the entire image. Default: 0.05.
        residual_reg_factor (float, optional): Loss weight for residual magnitude constraint.
            Prevents over-correction. Default: 0.005.
        refine_smooth_factor (float, optional): Loss weight for gradient smoothness in
            refined regions. Default: 0.0.
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
        refine_hidden=24,
        refine_use_backbone_feat=False,
        mask_sparse_factor=0.05,
        residual_reg_factor=0.005,
        refine_smooth_factor=0.0,
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

        # Spatial refinement head (optional)
        self.en_refine = en_refine
        if en_refine:
            self.refine_head = SpatialRefineHead(
                in_channels=6,
                hidden=refine_hidden,
                use_backbone_feat=refine_use_backbone_feat,
                backbone_feat_dim=self.backbone.out_channels
            )

        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.en_adaint = en_adaint
        self.sparse_factor = sparse_factor
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor
        self.refine_use_backbone_feat = refine_use_backbone_feat
        self.mask_sparse_factor = mask_sparse_factor
        self.residual_reg_factor = residual_reg_factor
        self.refine_smooth_factor = refine_smooth_factor
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
        self.adaint_fixed = False
        self.refine_fixed = False
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
            tuple(Tensor, Tensor, Tensor, Tensor or None):
                Output image, LUT weights, Sampling Coordinates, Refinement mask (if enabled).
        """
        # E: (b, f)
        codes = self.backbone(imgs)
        # (b, m), T: (b, c, d, d, d)
        weights, luts = self.lut_generator(codes)
        # \hat{P}: (b, c, d)
        if self.en_adaint:
            vertices = self.adaint(codes)
        else:
            vertices = self.uniform_vertices

        lut_out = ailut_transform(imgs, luts, vertices)

        # Spatial refinement (optional)
        refine_mask = None
        refine_residual = None
        if self.en_refine:
            backbone_feat = codes if self.refine_use_backbone_feat else None
            outs, refine_mask, refine_residual = self.refine_head(lut_out, imgs, backbone_feat)
        else:
            outs = lut_out

        return outs, weights, vertices, refine_mask, refine_residual

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
        output, weights, vertices, refine_mask, refine_residual = self.forward_dummy(lq)

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

        # Refinement losses (only when refinement is enabled)
        if self.en_refine and refine_residual is not None:
            # Debug: print every 100 iters
            iter_count = int(self.cnt_iters.item())
            if iter_count % 100 == 0:
                print(f"[DEBUG iter={iter_count}] residual: min={refine_residual.min().item():.6f}, max={refine_residual.max().item():.6f}, abs_mean={refine_residual.abs().mean().item():.6f}")

            # Mask sparsity (disabled by default now)
            if self.mask_sparse_factor > 0:
                losses['loss_mask_sparse'] = self.mask_sparse_factor * refine_mask.mean()
            
            # Residual magnitude constraint
            if self.residual_reg_factor > 0:
                losses['loss_residual_reg'] = self.residual_reg_factor * refine_residual.abs().mean()

            # Gradient smoothness in refined regions
            if self.refine_smooth_factor > 0:
                dx = output[..., :, 1:] - output[..., :, :-1]
                dy = output[..., 1:, :] - output[..., :-1, :]
                mask_x = refine_mask[..., :, 1:]
                mask_y = refine_mask[..., 1:, :]
                losses['loss_refine_smooth'] = self.refine_smooth_factor * (
                    (dx.abs() * mask_x).mean() + (dy.abs() * mask_y).mean())

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
        output, _, _, _, _ = self.forward_dummy(lq)
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

        # fix Refinement head in the first several epochs (let LUT learn first)
        if self.en_refine and self.cnt_iters < self.n_fix_refine_iters:
            if not self.refine_fixed:
                self.refine_fixed = True
                self.refine_head.requires_grad_(False)
                get_root_logger().info(f'Fix RefineHead for {self.n_fix_refine_iters} iters.')
        elif self.en_refine and self.cnt_iters == self.n_fix_refine_iters:
            self.refine_head.requires_grad_(True)
            if self.refine_fixed:
                self.refine_fixed = False
                get_root_logger().info(f'Unfix RefineHead after {self.n_fix_refine_iters} iters.')

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

        # 直接用 tensor 计算 PSNR，保持 float32 精度，不量化到 8-bit
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
                # SSIM 仍用 8-bit，因为原实现依赖 255 范围
                eval_result['SSIM'] = self.allowed_metrics['SSIM'](
                    tensor2img(output), tensor2img(gt), crop_border)
        return eval_result
