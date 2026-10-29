#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.general_utils import build_rotation

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def ray_elongation_loss(means, scales, rotations, cam_center, max_ratio=3.0):
    # Penalizes each Gaussian's elongation specifically ALONG its own
    # camera-ray direction - the one axis narrow-baseline capture leaves
    # genuinely unconstrained (see the 360geer densification investigation:
    # under weak parallax, the optimizer's cheapest way to explain an
    # ambiguous-depth pixel is to stretch a Gaussian along the ray so it
    # looks right from every training camera at once, then falls apart from
    # a held-out angle). Unlike lambda_aniso (arguments/__init__.py, penalizes
    # max/min scale ratio regardless of orientation), this only fires on
    # elongation aligned with the ray - a genuinely thin sideways structure
    # (a wire, a flat surface edge-on to the camera) has its long axis
    # perpendicular to the ray, not along it, so it's untouched.
    #
    # cam_center is a single fixed point (the training-camera centroid, see
    # train.py) rather than the per-iteration viewpoint - a deliberate
    # simplification valid specifically because this investigation's
    # problem scenes have near-co-located cameras (a few cm of baseline
    # total, see the camera-baseline survey); for a wide-baseline capture
    # this approximation would need revisiting to use the actual per-view
    # camera position instead.
    R = build_rotation(rotations)                       # (N,3,3)
    Sigma = R @ torch.diag_embed(scales ** 2) @ R.transpose(1, 2)
    v = F.normalize(means - cam_center, dim=-1)          # ray direction per Gaussian
    var_along = torch.einsum('ni,nij,nj->n', v, Sigma, v)
    var_perp = (Sigma.diagonal(dim1=1, dim2=2).sum(-1) - var_along) / 2
    ratio = torch.sqrt(var_along / (var_perp + 1e-8))
    return F.relu(ratio - max_ratio).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        masked_ssim = (ssim_map * mask).sum() / mask.sum()
        return masked_ssim

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

