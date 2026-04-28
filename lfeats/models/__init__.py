# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio feature extraction models."""

from .contentvec import ContentVecModel
from .data2vec import Data2VecModel
from .data2vec2 import Data2Vec2Model
from .ecapa_tdnn import EcapaTDNNModel
from .emotion2vec import Emotion2VecModel
from .emotion2vec_plus import Emotion2VecPlusModel
from .hubert import HuBERTModel
from .manager import ModelManager
from .next_tdnn import NeXtTDNNModel
from .r_spin import RSpinModel
from .r_vector import RVectorModel
from .redimnet import ReDimNetModel
from .spidr import SpidRModel
from .spin import SpinModel
from .sslzip import SSLZipModel
from .unispeech_sat import UniSpeechSATModel
from .wav2vec2 import Wav2Vec2Model
from .wavlm import WavLMModel
from .whisper import WhisperModel
from .x_vector import XVectorModel

MODEL_MAP = {
    "contentvec": ContentVecModel,
    "data2vec": Data2VecModel,
    "data2vec2": Data2Vec2Model,
    "ecapa-tdnn": EcapaTDNNModel,
    "emotion2vec": Emotion2VecModel,
    "emotion2vec+": Emotion2VecPlusModel,
    "hubert": HuBERTModel,
    "next-tdnn": NeXtTDNNModel,
    "r-spin": RSpinModel,
    "r-vector": RVectorModel,
    "redimnet": ReDimNetModel,
    "spidr": SpidRModel,
    "spin": SpinModel,
    "sslzip": SSLZipModel,
    "unispeech-sat": UniSpeechSATModel,
    "wav2vec2": Wav2Vec2Model,
    "wavlm": WavLMModel,
    "whisper": WhisperModel,
    "x-vector": XVectorModel,
}

__all__ = [
    "ModelManager",
    "MODEL_MAP",
]
