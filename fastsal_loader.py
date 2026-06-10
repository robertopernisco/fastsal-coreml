"""Carica FastSal (PyTorch) sulla torchvision corrente via compat-shim + preprocessing.

FastSal è scritto per una torchvision vecchia: 3 punti rotti su 0.24
(`InvertedResidual`/`ConvBNReLU` spostati, `ops.misc.Conv2d` rimosso). Gli shim qui
sotto li riallineano — i pesi pre-addestrati `salicon_A.pth` caricano con
missing=0/unexpected=0 (verificato), quindi il modello è fine-tunabile.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

FASTSAL_DIR = Path(os.environ.get(
    "FASTSAL_DIR", str(Path.home() / ".claude/scripts/saliency_test/FastSal")))

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def install_shims() -> None:
    """Rende importabile FastSal su torchvision moderna."""
    import torchvision.models.mobilenet as mb
    import torchvision.ops.misc as miscops
    from torchvision.models.mobilenetv2 import InvertedResidual
    mb.InvertedResidual = InvertedResidual
    if not hasattr(mb, "ConvBNReLU"):
        class ConvBNReLU(nn.Sequential):
            def __init__(self, i, o, kernel_size=3, stride=1, groups=1, norm_layer=None):
                p = (kernel_size - 1) // 2
                nl = norm_layer or nn.BatchNorm2d
                super().__init__(
                    nn.Conv2d(i, o, kernel_size, stride, p, groups=groups, bias=False),
                    nl(o), nn.ReLU6(inplace=True))
        mb.ConvBNReLU = ConvBNReLU
    if not hasattr(miscops, "Conv2d"):
        miscops.Conv2d = nn.Conv2d


def load_fastsal(weights: str = "salicon_A", model_type: str = "A",
                 device: str = "cpu", from_checkpoint: str | None = None):
    """Ritorna il modello FastSal (init salicon_A, o un checkpoint FT)."""
    install_shims()
    if str(FASTSAL_DIR) not in sys.path:
        sys.path.insert(0, str(FASTSAL_DIR))
    import model.fastSal as fastsal
    from utils import load_weight
    m = fastsal.fastsal(pretrain_mode=False, model_type=model_type)
    if from_checkpoint:
        m.load_state_dict(torch.load(from_checkpoint, map_location="cpu"))
    else:
        sd, _ = load_weight(str(FASTSAL_DIR / f"weights/{weights}.pth"), remove_decoder=False)
        m.load_state_dict(sd)
    return m.to(device)


def preprocess_np(path: str) -> np.ndarray:
    """Replica read_vgg_img: RGB → 192×256 → /255 → ImageNet-norm → [3,192,256]."""
    img = Image.open(path).convert("RGB").resize((256, 192), Image.LANCZOS)
    arr = (np.asarray(img, np.float32) / 255.0 - _MEAN) / _STD
    return arr.transpose(2, 0, 1).astype(np.float32)
