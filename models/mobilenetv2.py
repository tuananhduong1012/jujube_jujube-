"""
models/mobilenet_v2.py
──────────────────────
Custom MobileNetV2 with a two-layer classifier head.

Usage:
    python benchmark.py --model mobilenet_v2 --dataset ./data/jujube_have_flaws/dataset ...
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def build_model(num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model   = mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model
