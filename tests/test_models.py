# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the models within the Extractor class."""

import multiprocessing as mp

import numpy as np
import pytest
import torch

from lfeats import Extractor, Features
from tests.utils import generate_dummy_waveform


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("contentvec", "hubert-100"),
        ("dacvae", "base"),
        ("data2vec", "base"),
        ("data2vec2", "base"),
        ("ecapa-tdnn", "base"),
        ("emotion2vec", "base"),
        ("emotion2vec+", "base"),
        ("higgs-audio", "v2"),
        ("hubert", "base"),
        ("mimi", "base"),
        ("next-tdnn", "light"),
        ("r-spin", "wavlm-32"),
        ("r-vector", "base"),
        ("redimnet", "b0"),
        ("redimnet2", "b0"),
        ("spidr", "base"),
        ("spin", "hubert-128"),
        ("spin", "wavlm-128"),
        ("sslzip", "tiny"),
        ("unispeech-sat", "base"),
        ("w2v-bert2", "base"),
        ("wav2vec2", "base"),
        ("wavlm", "base"),
        ("wavlm-sv", "base"),
        ("whisper", "tiny"),
        ("x-codec", "hubert"),
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

    audio, sr = generate_dummy_waveform(1.0)
    features = extractor(audio, sr)
    assert isinstance(features, Features)
    B, _, _ = features.shape
    assert B == 1


@pytest.mark.parametrize(
    ("model_name", "variant", "sample_rate", "frame_shift"),
    [
        ("dacvae", "base", 48000, 1920),
        ("mimi", "base", 24000, 1920),
        ("w2v-bert2", "base", 16000, 320),
    ],
)
def test_frame_center(
    model_name: str, variant: str, sample_rate: int, frame_shift: int
) -> None:
    """Test if the n-th frame is centered at the n-th frame shift.

    An impulse is swept around the n-th frame, and the center of the range where the
    n-th frame is the most affected one is compared with the n-th frame shift. The
    first layer is used since the alignment is determined by the front-end.

    """
    extractor = Extractor(model_name, variant)
    extractor.load(quiet=True)

    audio, sr = generate_dummy_waveform(5.0, sample_rate=sample_rate)
    base = extractor(audio, sr, layers=0).array[0]
    impulse = np.hanning(9)

    def most_affected_frame(position: int) -> int:
        perturbed = audio.copy()
        perturbed[position - 4 : position + 5] += impulse
        diff = extractor(perturbed, sr, layers=0).array[0] - base
        return int(np.argmax(np.linalg.norm(diff, axis=-1)))

    n = 30
    step = frame_shift // 16
    offsets = [
        offset
        for offset in range(-frame_shift, frame_shift + 1, step)
        if most_affected_frame(n * frame_shift + offset) == n
    ]
    assert 0 < len(offsets)
    center = (offsets[0] + offsets[-1]) / 2
    assert abs(center) <= frame_shift / 8


def _worker_load_model(model_name: str, variant: str, verbose: bool = False) -> None:
    extractor = Extractor(model_name, variant, device="cpu")
    try:
        if verbose:
            print("Loading model in parallel...")
        extractor.load(quiet=not verbose)
        if verbose:
            print("Model loaded successfully in parallel.")
    except Exception as e:
        raise RuntimeError(f"Failed to load the model in parallel: {e}") from e


@pytest.mark.parametrize(
    ("model_name", "variant"),
    [
        ("contentvec", "hubert-100"),
        ("ecapa-tdnn", "base"),
        ("emotion2vec", "base"),
        ("hubert", "base"),
        ("spidr", "base"),
        ("spin", "wavlm-128"),
        ("sslzip", "tiny"),
    ],
)
def test_parallel_loading(model_name: str, variant: str, verbose: bool = False) -> None:
    """Test if the model can be loaded in parallel without errors."""
    ctx = mp.get_context("spawn")
    num_processes = 2
    tasks = [(model_name, variant, verbose) for _ in range(num_processes)]

    with ctx.Pool(processes=num_processes) as pool:
        try:
            pool.starmap_async(_worker_load_model, tasks).get()
        except Exception as e:
            raise RuntimeError(f"Failed to load the model in parallel: {e}") from e
