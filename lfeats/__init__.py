# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The main module for the audio feature extraction package."""

from .interfaces.extractor import Extractor
from .interfaces.resampler import Resampler
from .interfaces.types import Audio, Container, Features
from .version import __version__

__all__ = [
    "Audio",
    "Container",
    "Extractor",
    "Features",
    "Resampler",
    "__version__",
]
