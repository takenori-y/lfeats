# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the command-line interface."""

import os
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
        assert data["features"].dtype == np.float32

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
    assert data["features"].dtype == torch.float32
    length = data["features"].shape[-2]

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
    assert data.shape[-2] == length


def test_subdir_structure(monkeypatch) -> None:
    """Test the subdir structure in the output directory."""
    output_dir = "tests/outputs"

    output_file = f"{output_dir}/wav/noise.npz"
    if os.path.exists(output_file):
        os.remove(output_file)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lfeats",
            "tests/data/wav",
            "--output_dir",
            output_dir,
            "--output_format",
            "npz",
            "--subdir_offset",
            "2",
            "--quiet",
        ],
    )
    cli()

    with np.load(output_file) as data:
        assert "features" in data
