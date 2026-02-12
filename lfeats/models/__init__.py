# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio feature extraction models."""

from .base import Audio, Features
from .spin import SpinModel

MODEL_MAP = {
    "spin": SpinModel,
}

__all__ = [
    "Audio",
    "Features",
    "MODEL_MAP",
]
