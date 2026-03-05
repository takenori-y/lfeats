# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the models within the Extractor class."""

import pytest
import torch

from lfeats import Extractor, Features
from tests.utils import generate_dummy_waveform


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("contentvec", "hubert-100"),
        ("data2vec", "base"),
        ("data2vec2", "base"),
        ("ecapa-tdnn", "base"),
        ("emotion2vec", "base"),
        ("emotion2vec+", "base"),
        ("hubert", "base"),
        ("next-tdnn", "light"),
        ("r-spin", "wavlm-32"),
        ("r-vector", "base"),
        ("spidr", "base"),
        ("spin", "hubert-128"),
        ("spin", "wavlm-128"),
        ("sslzip", "tiny"),
        ("unispeech-sat", "base"),
        ("wav2vec2", "base"),
        ("wavlm", "base"),
        ("whisper", "tiny"),
        ("x-vector", "base"),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_running(model_name: str, variant: str, device: str) -> None:
    """Test if the model can run without errors."""
    extractor = Extractor(model_name, variant, device=device)
    extractor.load(quiet=True)

    audio, sr = generate_dummy_waveform(1)
    features = extractor(audio, sr)
    assert isinstance(features, Features)
    B, T, D = features.shape
    assert B == 1
    assert T == 1 or T == 50
