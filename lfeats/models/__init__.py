# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio feature extraction models."""

from .contentvec import ContentVecModel
from .hubert import HubertModel
from .manager import ModelManager
from .rspin import RSpinModel
from .spin import SpinModel
from .sslzip import SSLZipModel
from .wavlm import WavLMModel
from .whisper import WhisperModel

MODEL_MAP = {
    "contentvec": ContentVecModel,
    "hubert": HubertModel,
    "rspin": RSpinModel,
    "spin": SpinModel,
    "sslzip": SSLZipModel,
    "wavlm": WavLMModel,
    "whisper": WhisperModel,
}

__all__ = [
    "ModelManager",
    "MODEL_MAP",
]
