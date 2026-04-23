# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the models within the Extractor class."""

import multiprocessing as mp

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
        ("redimnet", "b0"),
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
