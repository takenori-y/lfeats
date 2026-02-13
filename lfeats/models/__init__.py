# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio feature extraction models."""

from .manager import ModelManager
from .spin import SpinModel

MODEL_MAP = {
    "spin": SpinModel,
}

__all__ = [
    "ModelManager",
    "MODEL_MAP",
]
