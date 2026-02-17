# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio resamplers."""

from .lilfilter import LilFilterResampler
from .manager import ResamplerManager
from .torchaudio import TorchAudioResampler

RESAMPLER_MAP = {
    "lilfilter": LilFilterResampler,
    "torchaudio": TorchAudioResampler,
}

__all__ = [
    "ResamplerManager",
    "RESAMPLER_MAP",
]
