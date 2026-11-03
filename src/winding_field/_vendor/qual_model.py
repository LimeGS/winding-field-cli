"""AFV V2 inference with segmentation, direction, centre and affinity outputs."""
import importlib
import json
from pathlib import Path

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class FiberDetector(nn.Module):
    def __init__(self, plans):
        super().__init__()
        architecture = plans["configurations"]["3d_fullres"]["architecture"]
        args = dict(architecture["arch_kwargs"])
        for key in architecture["_kw_requires_import"]:
            if args[key] is not None:
                module, name = args[key].rsplit(".", 1)
                args[key] = getattr(importlib.import_module(module), name)
        self.backbone = ResidualEncoderUNet(
            input_channels=1, num_classes=4, deep_supervision=False, **args
        )
        channels = args["features_per_stage"][0]
        self.centre = nn.Conv3d(channels, 1, 1)
        self.affinity = nn.Conv3d(channels, 6, 1)
        self.direction = nn.Conv3d(channels, 3, 1)

    def forward(self, x):
        """Accept normalized CT [B, 1, Z, Y, X]; return logits and XYZ axes."""
        skips = self.backbone.encoder(x)
        decoder = self.backbone.decoder
        x = skips[-1]
        for i, (up, stage) in enumerate(zip(decoder.transpconvs, decoder.stages)):
            x = stage(torch.cat((up(x), skips[-i-2]), dim=1))
        return {
            "seg": decoder.seg_layers[-1](x),
            "centre": self.centre(x),
            "affinity": self.affinity(x),
            "direction": self.direction(x),
        }


def normalize_ct(ct, mean=None, std=None):
    """Normalize uint8-scale CT, using region statistics when provided."""
    ct = ct.float()
    if mean is None:
        mean = ct.mean(dim=(2,3,4), keepdim=True)
    else:
        mean = torch.as_tensor(mean, device=ct.device, dtype=ct.dtype)
        if mean.ndim == 1:
            mean = mean[:,None,None,None,None]
    if std is None:
        std = ct.std(dim=(2,3,4), keepdim=True)
    else:
        std = torch.as_tensor(std, device=ct.device, dtype=ct.dtype)
        if std.ndim == 1:
            std = std[:,None,None,None,None]
    return (ct - mean) / std.clamp_min(10.0)


def load_model(folder, device="cpu"):
    """Load all heads from this repository's inference checkpoint."""
    folder = Path(folder)
    model = FiberDetector(json.loads((folder / "plans.json").read_text()))
    checkpoint = torch.load(folder / "checkpoint_final.pth", map_location="cpu", weights_only=True)
    state = {"backbone." + k: v for k,v in checkpoint["network_weights"].items()}
    state.update(checkpoint["auxiliary_weights"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
