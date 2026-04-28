# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the Resampler class."""

import pytest
import torch

from lfeats import Resampler
from tests.utils import generate_dummy_waveform


@pytest.mark.parametrize(
    ("resampler_type", "resampler_preset"),
    [
        ("lilfilter", "base"),
        ("torchaudio", "kaiser-fast"),
    ],
)
def test_device(
    resampler_type: str,
    resampler_preset: str | None,
    src_rate: int = 44100,
    dst_rate: int = 16000,
) -> None:
    """Test if the resampler can be moved to the specified device."""
    resampler = Resampler(resampler_type, resampler_preset, device="cpu")

    audio, sr = generate_dummy_waveform(1, sample_rate=src_rate)
    resampled_audio = resampler(audio, src_rate=sr, dst_rate=dst_rate)
    assert resampled_audio.tensor.device.type == "cpu"

    if torch.cuda.is_available():
        resampler.to("cuda")
        resampled_audio = resampler(audio, src_rate=sr, dst_rate=dst_rate)
        assert resampled_audio.tensor.device.type == "cuda"

        resampler = Resampler(resampler_type, resampler_preset, device="cuda")
        resampled_audio = resampler(audio, src_rate=sr, dst_rate=dst_rate)
        assert resampled_audio.tensor.device.type == "cuda"

        resampler.to("cpu")
        resampled_audio = resampler(audio, src_rate=sr, dst_rate=dst_rate)
        assert resampled_audio.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("resampler_type", "resampler_preset"),
    [
        ("lilfilter", "base"),
        ("scipy", "fft"),
        ("scipy", "poly"),
        ("soxr", "quick"),
        ("torchaudio", "kaiser-fast"),
        ("torchaudio", "kaiser-best"),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_running(
    resampler_type: str,
    resampler_preset: str | None,
    device: str,
    src_rate: int = 44100,
    dst_rate: int = 16000,
    verbose: bool = False,
) -> None:
    """Test if the resampler can run without errors."""
    resampler = Resampler(resampler_type, resampler_preset, device=device)

    audio, sr = generate_dummy_waveform(1, sample_rate=src_rate)
    resampled_audio = resampler(audio, src_rate=sr, dst_rate=dst_rate)
    assert resampled_audio.length == dst_rate

    if verbose:
        resampled_audio.tofile(f"resampled_{resampler_type}_{resampler_preset}.wav")
