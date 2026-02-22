# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio feature extraction models."""

from .contentvec import ContentVecModel
from .data2vec import Data2VecModel
from .data2vec2 import Data2Vec2Model
from .hubert import HubertModel
from .manager import ModelManager
from .r_spin import RSpinModel
from .spidr import SpidRModel
from .spin import SpinModel
from .sslzip import SSLZipModel
from .unispeech_sat import UniSpeechSatModel
from .wav2vec2 import Wav2Vec2Model
from .wavlm import WavLMModel
from .whisper import WhisperModel

MODEL_MAP = {
    "contentvec": ContentVecModel,
    "data2vec": Data2VecModel,
    "data2vec2": Data2Vec2Model,
    "hubert": HubertModel,
    "r-spin": RSpinModel,
    "spidr": SpidRModel,
    "spin": SpinModel,
    "sslzip": SSLZipModel,
    "unispeech-sat": UniSpeechSatModel,
    "wav2vec2": Wav2Vec2Model,
    "wavlm": WavLMModel,
    "whisper": WhisperModel,
}

__all__ = [
    "ModelManager",
    "MODEL_MAP",
]
