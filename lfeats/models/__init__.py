# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio feature extraction models."""

from .hubert import HubertModel
from .manager import ModelManager
from .spin import SpinModel
from .whisper import WhisperModel

MODEL_MAP = {
    "hubert": HubertModel,
    "spin": SpinModel,
    "whisper": WhisperModel,
}

__all__ = [
    "ModelManager",
    "MODEL_MAP",
]
