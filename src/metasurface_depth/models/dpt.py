# Copyright (c) 2024, Depth Anything V2
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import _make_scratch, _make_fusion_block
from .resnet import ResNetEncoder

class InverseShift(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return 1.0 / (x + self.epsilon)

class DPTHead(nn.Module):
    def __init__(self,
                 nclass,
                 in_channels,
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 output_act='sigmoid',
                 prompt_channels=2,
                 resnet_enabled=False,
                 resnet_blocks_per_stage=3):
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.resnet_enabled = resnet_enabled

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        if resnet_enabled:
            self.scratch.depth_encoder = _make_scratch(
                out_channels,
                features,
                groups=1,
                expand=False,
            )
            self.scratch.depth_encoder.encoder = ResNetEncoder(
                in_channels=prompt_channels,
                blocks_per_stage=resnet_blocks_per_stage,
                out_channels=out_channels
            )
        else:
            # If ResNet is not enabled, create a simple fusion block
            self.scratch.depth_extractor = _make_fusion_block(
                prompt_channels, features, use_bn, first_layer=True)

        self.scratch.refinenet1 = _make_fusion_block(
            features, features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(
            features, features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(
            features, features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(
            features, features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        if output_act == 'sigmoid':
            act_func = nn.Sigmoid()
        elif output_act == 'inverse':
            act_func = InverseShift()
        else:
            act_func = nn.Identity()

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass,
                          kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1,
                          stride=1, padding=0),
                act_func,
            )

    def forward(self, out_features, patch_h, patch_w, prompt=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        if self.resnet_enabled:
            prompt_features = self.scratch.depth_encoder.encoder(prompt)
            pf_1_rn = self.scratch.depth_encoder.layer1_rn(prompt_features[0])
            pf_2_rn = self.scratch.depth_encoder.layer2_rn(prompt_features[1])
            pf_3_rn = self.scratch.depth_encoder.layer3_rn(prompt_features[2])
            pf_4_rn = self.scratch.depth_encoder.layer4_rn(prompt_features[3])

            path_4 = self.scratch.refinenet4(
                layer_4_rn, size=layer_3_rn.shape[2:], prompt_depth=pf_4_rn)
            path_3 = self.scratch.refinenet3(
                path_4, layer_3_rn, size=layer_2_rn.shape[2:], prompt_depth=pf_3_rn)
            path_2 = self.scratch.refinenet2(
                path_3, layer_2_rn, size=layer_1_rn.shape[2:], prompt_depth=pf_2_rn)
            path_1 = self.scratch.refinenet1(
                path_2, layer_1_rn, prompt_depth=pf_1_rn)
        else:
            path_0 = self.scratch.depth_extractor(
                prompt, size=prompt.shape[-2:], prompt_depth=prompt)
            # 896 -> 768x32 384x64 192x128 96x256
            path_4 = self.scratch.refinenet4(
                layer_4_rn, size=layer_3_rn.shape[2:], prompt_depth=path_0)
            path_3 = self.scratch.refinenet3(
                path_4, layer_3_rn, size=layer_2_rn.shape[2:], prompt_depth=path_0)
            path_2 = self.scratch.refinenet2(
                path_3, layer_2_rn, size=layer_1_rn.shape[2:], prompt_depth=path_0)
            path_1 = self.scratch.refinenet1(
                path_2, layer_1_rn, prompt_depth=path_0)

        out = self.scratch.output_conv1(path_1)
        out_feat = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out_feat)
        return out
