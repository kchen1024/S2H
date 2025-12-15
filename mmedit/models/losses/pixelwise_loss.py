# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import masked_loss

_reduction_modes = ['none', 'mean', 'sum']


@masked_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


@masked_loss
def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps)


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class MaskedTVLoss(L1Loss):
    """Masked TV loss.

        Args:
            loss_weight (float, optional): Loss weight. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def forward(self, pred, mask=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor, optional): Tensor with shape of (n, 1, h, w).
                Defaults to None.

        Returns:
            [type]: [description]
        """
        y_diff = super().forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :], weight=mask[:, :, :-1, :])
        x_diff = super().forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:], weight=mask[:, :, :, :-1])

        loss = x_diff + y_diff

        return loss


@LOSSES.register_module()
class ColorGamutLoss(nn.Module):
    """Color gamut matching loss using histogram statistics.

    Encourages the predicted image to match the color distribution
    of the ground truth HDR image, helping expand the color gamut.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        num_bins (int): Number of histogram bins. Default: 64.
        color_space (str): Color space for comparison ('rgb', 'hsv', 'lab').
            Default: 'rgb'.
    """

    def __init__(self, loss_weight=1.0, num_bins=64, color_space='rgb'):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_bins = num_bins
        self.color_space = color_space.lower()
        assert self.color_space in ['rgb', 'hsv', 'lab'], \
            f'Unsupported color space: {self.color_space}. ' \
            f'Supported ones are: rgb, hsv, lab'

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (Tensor): Predicted tensor, shape (n, c, h, w).
            target (Tensor): Ground truth tensor, shape (n, c, h, w).

        Returns:
            Tensor: Color gamut loss.
        """
        # Convert to appropriate color space if needed
        if self.color_space == 'hsv':
            pred = self._rgb_to_hsv(pred)
            target = self._rgb_to_hsv(target)
        elif self.color_space == 'lab':
            pred = self._rgb_to_lab(pred)
            target = self._rgb_to_lab(target)

        # Compute per-channel histogram statistics
        loss = 0.0
        for c in range(pred.shape[1]):
            pred_channel = pred[:, c]
            target_channel = target[:, c]

            # Compute mean and std
            pred_mean = pred_channel.mean()
            target_mean = target_channel.mean()
            pred_std = pred_channel.std()
            target_std = target_channel.std()

            # Match first and second moments
            loss += F.l1_loss(pred_mean, target_mean)
            loss += F.l1_loss(pred_std, target_std)

        return loss * self.loss_weight

    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space.

        Args:
            rgb (Tensor): RGB tensor, shape (n, 3, h, w).

        Returns:
            Tensor: HSV tensor, shape (n, 3, h, w).
        """
        # Simplified HSV conversion for differentiability
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        max_rgb, _ = torch.max(rgb, dim=1)
        min_rgb, _ = torch.min(rgb, dim=1)
        diff = max_rgb - min_rgb

        # Value
        v = max_rgb

        # Saturation (add epsilon for numerical stability)
        s = diff / (max_rgb + 1e-7)

        # Hue (simplified)
        h = torch.zeros_like(v)

        return torch.stack([h, s, v], dim=1)

    def _rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space (approximate, differentiable).

        Args:
            rgb (Tensor): RGB tensor, shape (n, 3, h, w), values in [0, 1].

        Returns:
            Tensor: LAB tensor, shape (n, 3, h, w).
        """
        # RGB to XYZ (sRGB with D65 illuminant)
        rgb = torch.clamp(rgb, 1e-6, 1.0)

        # Linearize sRGB
        mask = rgb > 0.04045
        rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        r, g, b = rgb_linear[:, 0:1], rgb_linear[:, 1:2], rgb_linear[:, 2:3]

        # RGB to XYZ matrix (D65)
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        # Normalize by D65 white point
        x = x / 0.95047
        z = z / 1.08883

        # XYZ to LAB
        epsilon = 0.008856
        kappa = 903.3

        fx = torch.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
        fy = torch.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
        fz = torch.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b_ch = 200 * (fy - fz)

        return torch.cat([L, a, b_ch], dim=1)


@LOSSES.register_module()
class HDRToneLoss(nn.Module):
    """HDR tone mapping loss in log-luminance space.

    Computes loss in log-luminance space to better preserve HDR
    characteristics and handle the wide dynamic range.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        epsilon (float): Small constant for numerical stability. Default: 1e-3.
    """

    def __init__(self, loss_weight=1.0, epsilon=1e-3):
        super().__init__()
        self.loss_weight = loss_weight
        self.epsilon = epsilon

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (Tensor): Predicted tensor, shape (n, c, h, w).
            target (Tensor): Ground truth tensor, shape (n, c, h, w).

        Returns:
            Tensor: HDR tone mapping loss.
        """
        # Compute luminance (ITU-R BT.709)
        pred_lum = 0.2126 * pred[:, 0] + 0.7152 * pred[:, 1] + 0.0722 * pred[:, 2]
        target_lum = 0.2126 * target[:, 0] + 0.7152 * target[:, 1] + 0.0722 * target[:, 2]

        # Clamp luminance to positive values for numerical stability
        # HDR values can be > 1.0, so we only clamp the lower bound
        pred_lum = torch.clamp(pred_lum, min=self.epsilon)
        target_lum = torch.clamp(target_lum, min=self.epsilon)

        # Log-luminance space using log1p for better numerical stability
        # log1p(x) = log(1 + x), more stable for small values
        pred_log_lum = torch.log(pred_lum)
        target_log_lum = torch.log(target_lum)

        # L1 loss in log space
        loss = F.l1_loss(pred_log_lum, target_log_lum)

        return loss * self.loss_weight
