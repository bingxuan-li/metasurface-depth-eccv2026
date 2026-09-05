"""Fixed DAV2-derived architecture for the selected November checkpoints."""

import torch
from torch import nn
from ._vendor.vision_transformer import vit_small, vit_base, vit_large
from .models.dpt import DPTHead
from .models.config import model_configs

ENCODERS = {"small": "vits", "base": "vitb", "large": "vitl"}


class MetasurfaceDepth(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in ENCODERS:
            raise ValueError(f"Unknown variant: {variant}")
        encoder = ENCODERS[variant]
        cfg = model_configs[encoder]
        factories = {"vits": vit_small, "vitb": vit_base, "vitl": vit_large}
        # Exact defaults from the experiment's local hubconf; no network downloads.
        self.pretrained = factories[encoder](
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        self.depth_head = DPTHead(
            nclass=1,
            in_channels=self.pretrained.blocks[0].attn.qkv.in_features,
            features=cfg["features"],
            out_channels=cfg["out_channels"],
            use_bn=False,
            use_clstoken=False,
            output_act="identity",
            prompt_channels=3,
            resnet_enabled=False,
        )
        self.layer_idxs = cfg["layer_idxs"]
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pseudo_rgb):
        h, w = pseudo_rgb.shape[-2:]
        if h < 14 or w < 14 or h % 14 or w % 14:
            raise ValueError("Input height and width must be positive multiples of 14")
        features = self.pretrained.get_intermediate_layers(
            (pseudo_rgb - self._mean) / self._std, self.layer_idxs, return_class_token=True
        )
        # Zero-valued prompt is intentional: it reproduces train.forward(mode='rgb').
        return self.depth_head(features, h // 14, w // 14, torch.zeros_like(pseudo_rgb))


def load_model(weights, device="cpu", expected_variant=None):
    payload = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("format_version") != 1:
        raise ValueError("Expected exported format_version=1 weights; run export first")
    config = payload["config"]
    if expected_variant and config["variant"] != expected_variant:
        raise ValueError(f"Checkpoint is {config['variant']}, requested {expected_variant}")
    model = MetasurfaceDepth(config["variant"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), config
