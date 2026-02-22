# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the command-line interface."""

import sys

import numpy as np
import torch

from lfeats.cli import main as cli


def test_running(monkeypatch) -> None:
    """Test if the CLI can run without errors."""
    output_dir = "tests/outputs"

    # File input and NPZ output
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lfeats",
            "tests/data/wav/noise.wav",
            "--output_dir",
            output_dir,
            "--output_format",
            "npz",
            "--quiet",
        ],
    )
    cli()

    with np.load(f"{output_dir}/noise.npz") as data:
        assert "features" in data

    # Directory input and PT output
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lfeats",
            "tests/data/wav",
            "--output_dir",
            output_dir,
            "--output_format",
            "pt",
            "--quiet",
        ],
    )
    cli()

    data = torch.load(f"{output_dir}/noise.pt")
    assert "features" in data

    # SCP input and binary output
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lfeats",
            "tests/data/input.scp",
            "--output_dir",
            output_dir,
            "--output_format",
            "float",
            "--quiet",
        ],
    )
    cli()

    data = np.fromfile(f"{output_dir}/noise.feats", dtype=np.float32).reshape(-1, 768)
