# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the Extractor class."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from lfeats import Extractor, Features
from tests.utils import generate_dummy_waveform


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("spin", "hubert-128"),
        ("spin", "wavlm-128"),
        ("whisper", "tiny"),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_running(model_name: str, variant: str, device: str) -> None:
    """Test if the model can run without errors."""
    extractor = Extractor(model_name, variant, device=device)

    audio, sr = generate_dummy_waveform(1)
    features = extractor(audio, sr)
    assert isinstance(features, Features)


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("spin", "hubert-128"),
    ],
)
def test_chunking(model_name: str, variant: str, verbose: bool = False) -> None:
    """Test for the effect of chunking on the extracted features."""
    extractor = Extractor(model_name, variant)

    audio, sr = generate_dummy_waveform(10.01)
    org_features = extractor(audio, sr)

    features1 = extractor(audio, sr, chunk_length_sec=5, overlap_length_sec=4)
    features2 = extractor(audio, sr, chunk_length_sec=5, overlap_length_sec=0)

    error1 = np.abs(features1.array - org_features.array)[0]
    error2 = np.abs(features2.array - org_features.array)[0]
    assert error1.mean() < error2.mean()

    if verbose:
        plt.imsave(f"{model_name}_chunking_error1.png", error1.T)
        plt.imsave(f"{model_name}_chunking_error2.png", error2.T)


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("whisper", "tiny"),
    ],
)
def test_long_audio(model_name: str, variant: str) -> None:
    """Test if the model can process long audio without errors."""
    extractor = Extractor(model_name, variant)

    audio, sr = generate_dummy_waveform(60.01)
    features = extractor(audio, sr, overlap_length_sec=0)
    assert isinstance(features, Features)
