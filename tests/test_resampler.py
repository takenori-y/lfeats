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
        ("lilfilter", None),
        ("soxr", "quick"),
        ("torchaudio", "kaiser-fast"),
        ("torchaudio", "kaiser-best"),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_running(
    resampler_type: str, resampler_preset: str | None, device: str
) -> None:
    """Test if the resampler can run without errors."""
    resampler = Resampler(resampler_type, resampler_preset, device=device)

    audio, sr = generate_dummy_waveform(1, sample_rate=44100)
    resampled_audio = resampler(audio, src_rate=sr, dst_rate=16000)
    assert resampled_audio.length == 16000
