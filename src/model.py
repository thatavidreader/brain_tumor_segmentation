"""
Attention U-Net network builder.

Uses MONAI's built-in `AttentionUnet`, which implements attention gates
on the skip connections (Oktay et al., 2018) to let the decoder suppress
irrelevant background/edema regions and focus on tumor sub-structures.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn
from monai.networks.nets import AttentionUnet


def build_model(config: dict[str, Any]) -> nn.Module:
    m = config["model"]
    return AttentionUnet(
        spatial_dims=3,
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        channels=m["channels"],
        strides=m["strides"],
        dropout=m["dropout"],
    )
