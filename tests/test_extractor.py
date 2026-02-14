# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the Extractor class."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lfeats import Extractor, Features
from tests.utils import generate_dummy_waveform


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("spin", "hubert-128"),
        ("spin", "wavlm-128"),
    ],
)
def test_running(model_name: str, variant: str) -> None:
    """Test if the model can run without errors."""
    extractor = Extractor(model_name, variant)

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

    audio, sr = generate_dummy_waveform(10)
    org_features = extractor(audio, sr)

    features1 = extractor(audio, sr, chunk_length_sec=5, overlap_length_sec=4)
    features2 = extractor(audio, sr, chunk_length_sec=5, overlap_length_sec=0)

    error1 = np.abs(features1.array - org_features.array).mean()
    error2 = np.abs(features2.array - org_features.array).mean()
    assert error1 < error2

    if verbose:
        plt.imsave("chunking_error1.png", error1.T)
        plt.imsave("chunking_error2.png", error2.T)
