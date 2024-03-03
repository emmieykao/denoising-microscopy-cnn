"""
CREDITS TO: https://github.com/yinhaoz/denoising-fluorescence
"""

import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd

def cal_psnr(clean, noisy, max_val=255, normalized=False):
    """
    Args:
        clean (Tensor): [0, 255], BCHW
        noisy (Tensor): [0, 255], BCHW
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        PSNR per image: (B,)
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)
    mse = F.mse_loss(noisy, clean, reduction='none').view(clean.shape[0], -1).mean(1)
    return 10 * torch.log10(max_val ** 2 / mse)


def cal_ssim(clean, noisy, normalized=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM

    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)

    clean, noisy = clean.numpy(), noisy.numpy()
    ssim = np.array([structural_similarity(clean[i, 0], noisy[i, 0], data_range=255) 
        for i in range(clean.shape[0])])

    return ssim