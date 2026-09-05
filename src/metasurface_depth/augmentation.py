"""Original sensor augmentation math, isolated from legacy dataset paths."""

import random
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class Augmentation:
    def __init__(
        self,
        augment_prob,
        max_p_noise=0.0,
        max_g_noise=0.0,
        max_gs_blur=0.0,
        max_global_imbalance=0.0,
        max_local_imbalance=0.0,
        max_gs_kernels=0,
    ):
        self.ph_num = 1000
        self.augment_prob = augment_prob
        self.max_p_noise = max_p_noise
        self.max_g_noise = max_g_noise
        self.max_gs_blur = max_gs_blur
        self.max_global_imbalance = max_global_imbalance
        self.max_local_imbalance = max_local_imbalance
        self.max_gs_kernels = max_gs_kernels
        self.alpha = 1
        self.beta = 3

    def augment(self, img, gray=False):
        if random.random() > self.augment_prob:
            return img

        H, W = img.shape[-2:]
        if self.max_gs_blur > 0 and not gray > 0:
            max_gs_blur = self.max_gs_blur
            gs_blur = np.random.uniform(0, max_gs_blur)
            img = torch.from_numpy(gaussian_filter(img.numpy(), sigma=gs_blur)).to(img.dtype)

        # Add global imbalance to the output images
        if self.max_global_imbalance > 0:
            global_imbalance = np.random.beta(self.alpha, self.beta) * self.max_global_imbalance
            if np.random.random() < 0.5:
                global_imbalance = -global_imbalance
            img = img * (1 + global_imbalance)
        # Local imbalance
        if self.max_local_imbalance > 0 and self.max_gs_kernels > 0:
            mask = torch.zeros_like(img)
            y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            x_grid = x_grid.float()
            y_grid = y_grid.float()
            n_kernels = random.randint(1, self.max_gs_kernels)
            for _ in range(n_kernels):
                # Random center
                cx = random.uniform(0, W - 1)
                cy = random.uniform(0, H - 1)
                # Random anisotropic Gaussian parameters
                log_aspect_ratio = random.uniform(-1, 1)  # log(r), r in [0.1, 10]
                aspect_ratio = np.exp(log_aspect_ratio)  # r
                log_sigma_y = random.uniform(np.log(1), np.log(256))
                sigma_y = np.exp(log_sigma_y)
                sigma_x = sigma_y * aspect_ratio  # sigma_x = r * sigma_y
                theta = random.uniform(0, np.pi)  # orientation
                # Rotation matrix
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                # Apply rotation
                x_shift = x_grid - cx
                y_shift = y_grid - cy
                x_rot = cos_t * x_shift + sin_t * y_shift
                y_rot = -sin_t * x_shift + cos_t * y_shift
                gauss = torch.exp(-((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2) / 2)
                if np.random.random() < 0.5:
                    gauss = -gauss  # Randomly flip the sign
                mask += gauss
            if mask.abs().max() > 1e-6:
                mask = mask / mask.abs().max()  # Normalize the mask
            local_imbalance = np.random.beta(self.alpha, self.beta) * self.max_local_imbalance
            mask = mask * local_imbalance
            img = img * (1 + mask)
        img = torch.clamp(img, min=0)  # Ensure non-negative value
        # Add shot noise to the output images
        if self.max_p_noise > 0:
            min_noise = 1e-3 * self.max_p_noise  # avoid divide-by-zero or huge `times`
            p_noise = (
                np.random.beta(self.alpha, self.beta) * (self.max_p_noise - min_noise) + min_noise
            )
            times = self.ph_num / p_noise
            img = torch.poisson(img * times) / times
        # Add gaussian noise to the output images
        if self.max_g_noise > 0:
            g_noise = np.random.beta(self.alpha, self.beta) * self.max_g_noise
            img = img + torch.randn_like(img) * g_noise
        # ensure non-negative value
        img = torch.clamp(img, min=0, max=1)
        return img
