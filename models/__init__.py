"""
Model registry and shared output dataclass.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    depth:    Tensor                              # (B, S, C, 1, H, W) — always present
    semantic: Optional[Tensor] = None            # (B, S, C, 23, H, W) — joint models only
    state:    Optional[dict] = None              # stateful models only


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict = {}


def register(name: str):
    """Class decorator that registers a model under the given name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_models():
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# Import all model files to trigger registration
# ---------------------------------------------------------------------------

from . import (
    baseline_depth,
    baseline_seg_depth,
    video_seg_depth,
    video_seg_depth_resnet,
    video_former_depth,
    video_former_seg_depth,
)
