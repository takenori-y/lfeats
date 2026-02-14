# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""The main module for the audio feature extraction package."""

from .interfaces.extractor import Extractor as Extractor
from .interfaces.types import Audio as Audio
from .interfaces.types import Features as Features
from .version import __version__ as __version__
