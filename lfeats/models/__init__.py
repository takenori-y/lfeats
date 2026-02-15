# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio feature extraction models."""

from .hubert import HubertModel
from .manager import ModelManager
from .rspin import RSpinModel
from .spin import SpinModel
from .sslzip import SslZipModel
from .whisper import WhisperModel

MODEL_MAP = {
    "hubert": HubertModel,
    "rspin": RSpinModel,
    "spin": SpinModel,
    "sslzip": SslZipModel,
    "whisper": WhisperModel,
}

__all__ = [
    "ModelManager",
    "MODEL_MAP",
]
