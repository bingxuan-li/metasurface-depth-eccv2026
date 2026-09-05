"""Shared ordered-image and label contract for training and evaluation."""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .io import read_manifest


class EncodedDataset(Dataset):
    def __init__(self, manifest, crop_size=None, random_crop=False):
        self.rows = read_manifest(manifest, ("image1", "image2", "depth"))
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.augmentation = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        for key in ("image1", "image2"):
            raw = cv2.imread(str(row[key]), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint8:
                raise ValueError(f"{key} must be a readable uint8 image")
            images.append(cv2.imread(str(row[key]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255)
        target = np.load(row["depth"], allow_pickle=False)
        h, w = images[0].shape
        if images[1].shape != (h, w) or target.shape != (h, w):
            raise ValueError(
                "Encoded pair and metric depth must be aligned, with identical dimensions"
            )
        if target.dtype.kind != "f":
            raise ValueError("Depth must be floating point in meters")
        ch, cw = h // 14 * 14, w // 14 * 14
        if self.crop_size is not None:
            ch, cw = min(ch, self.crop_size), min(cw, self.crop_size)
        if min(ch, cw) < 14 or ch % 14 or cw % 14:
            raise ValueError("Crop dimensions must be positive multiples of 14")
        if self.random_crop:
            y, x = np.random.randint(h - ch + 1), np.random.randint(w - cw + 1)
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        a, b = [torch.from_numpy(im[y : y + ch, x : x + cw]) for im in images]
        if self.augmentation is not None:
            a, b = self.augmentation.augment(a), self.augmentation.augment(b)
        tensor = torch.stack((a, b, (a + b) / 2))
        label = torch.from_numpy(target[y : y + ch, x : x + cw].astype(np.float32)).unsqueeze(0)
        return tensor, label


def assert_disjoint(train_sets, validation_sets):
    """Reject duplicate input/label paths across train and validation splits."""
    training = {
        str(row[key])
        for dataset in train_sets
        for row in dataset.rows
        for key in ("image1", "image2", "depth")
    }
    validation = {
        str(row[key])
        for dataset in validation_sets
        for row in dataset.rows
        for key in ("image1", "image2", "depth")
    }
    if training & validation:
        raise ValueError("Train/validation overlap detected; use disjoint manifests")
