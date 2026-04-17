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
        ("hubert", "base"),
        ("ecapa-tdnn", "base"),
    ],
)
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_device(model_name: str, variant: str, sample_rate: int) -> None:
    """Test if the model can be moved to the specified device."""
    extractor = Extractor(model_name, variant, device="cpu")
    extractor.load(quiet=True)

    audio, sr = generate_dummy_waveform(1, sample_rate=sample_rate)
    features = extractor(audio, sr)
    assert features.tensor.device.type == "cpu"

    if torch.cuda.is_available():
        extractor.to("cuda")
        features = extractor(audio, sr)
        assert features.tensor.device.type == "cuda"

        extractor = Extractor("hubert", "base", device="cpu")
        extractor.to("cuda")
        extractor.load(quiet=True)
        features = extractor(audio, sr)
        assert features.tensor.device.type == "cuda"

        extractor.to("cpu")
        features = extractor(audio, sr)
        assert features.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("hubert", "base"),
    ],
)
@pytest.mark.parametrize("center", [True, False])
def test_chunking(
    model_name: str, variant: str, center: bool, verbose: bool = False
) -> None:
    """Test for the effect of chunking on the extracted features."""
    extractor = Extractor(model_name, variant)
    extractor.load(quiet=True)

    audio, sr = generate_dummy_waveform(10.01)
    org_features = extractor(audio, sr)

    features1 = extractor(
        audio, sr, center=center, chunk_length_sec=5, overlap_length_sec=4
    )
    features2 = extractor(
        audio, sr, center=center, chunk_length_sec=5, overlap_length_sec=0
    )

    error1 = np.abs(features1.array - org_features.array)[0]
    error2 = np.abs(features2.array - org_features.array)[0]
    if verbose:
        plt.imsave(f"tests/outputs/{model_name}_chunking_error1.png", error1.T)
        plt.imsave(f"tests/outputs/{model_name}_chunking_error2.png", error2.T)
    assert error1.mean() < error2.mean()


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("ecapa-tdnn", "base"),
        ("next-tdnn", "base"),
    ],
)
def test_reduction(model_name: str, variant: str) -> None:
    """Test for the reduction of features along the time axis."""
    extractor = Extractor(model_name, variant)
    extractor.load(quiet=True)

    audio, sr = generate_dummy_waveform(10, num_channels=2)
    features1 = extractor(
        audio, sr, chunk_length_sec=6, overlap_length_sec=0, reduction="mean"
    )
    assert features1.shape == (2, 1, 192)
    features2 = extractor(
        audio, sr, chunk_length_sec=6, overlap_length_sec=0, reduction="none"
    )
    assert features2.shape == (2, 2, 192)


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("hubert", "base"),
    ],
)
def test_upsampling(model_name: str, variant: str) -> None:
    """Test if the upsampling of features works correctly."""
    extractor = Extractor(model_name, variant)
    extractor.load(quiet=True)

    audio1, sr = generate_dummy_waveform(5)
    audio2 = np.pad(audio1[160:], (0, 160))

    features1 = extractor(audio1, sr, upsample_factor=1)
    features2 = extractor(audio2, sr, upsample_factor=1)
    upsampled_features = extractor(audio1, sr, upsample_factor=2)

    error1 = np.abs(features1.array - upsampled_features.array[:, 0::2])[0]
    error2 = np.abs(features2.array - upsampled_features.array[:, 1::2])[0]
    assert error1.sum() == 0
    assert error2.sum() == 0


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("whisper", "tiny"),
    ],
)
def test_long_audio(model_name: str, variant: str) -> None:
    """Test if the model can process long audio without errors."""
    extractor = Extractor(model_name, variant)
    extractor.load(quiet=True)

    audio, sr = generate_dummy_waveform(60.01)
    features = extractor(audio, sr, overlap_length_sec=0)
    assert isinstance(features, Features)
