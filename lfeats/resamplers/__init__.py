# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The module for audio resamplers."""

from .lilfilter import LilFilterResampler
from .manager import ResamplerManager
from .scipy import ScipyResampler
from .soxr import SoxrResampler
from .torchaudio import TorchAudioResampler

RESAMPLER_MAP = {
    "lilfilter": LilFilterResampler,
    "scipy": ScipyResampler,
    "soxr": SoxrResampler,
    "torchaudio": TorchAudioResampler,
}

__all__ = [
    "ResamplerManager",
    "RESAMPLER_MAP",
]
