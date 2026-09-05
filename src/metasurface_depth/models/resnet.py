import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=[96, 192, 384, 768], blocks_per_stage=2, num_stages=4, patch_size=14):
        """
        Args:
            in_channels (int): Input channel count (e.g., 3 for RGB).
            out_channels (list[int]): Feature dimensions for all stages.
            blocks_per_stage (int): Number of ResNet blocks between each maxpool.
            num_stages (int): Total number of downsampling stages.
        """
        super().__init__()
        # Target resolution *4 / 14, then /2 /4 /8
        oc = out_channels
        bps = blocks_per_stage
        self.patch_size = patch_size
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, oc[0], kernel_size=1, stride=1, padding=0),
            *[ResidualBlock(oc[0]) for _ in range(bps)],
            self.max_pool,
            *[ResidualBlock(oc[0]) for _ in range(bps)],
        )
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            sc = oc[i] if i == 0 else oc[i-1]
            stage = nn.Sequential(
                nn.Conv2d(sc, oc[i], kernel_size=1, stride=1, padding=0),
                *[ResidualBlock(oc[i]) for _ in range(bps)],
                self.max_pool
            )
            self.stages.append(stage)

    def forward(self, x):
        x = self.stem(x)
        x = F.interpolate(x, scale_factor=(16/self.patch_size, 16/self.patch_size), mode='bilinear')
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)  # store downsampled feature map
        return outputs  # list of feature maps at different resolutions