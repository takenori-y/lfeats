# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Utility functions for testing."""

import numpy as np


def generate_dummy_waveform(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    num_channels: int = 1,
    seed: int = 12345,
) -> tuple[np.ndarray, int]:
    """Generate a dummy waveform for testing.

    Parameters
    ----------
    duration_sec : float
        Duration of the waveform in seconds.

    sample_rate : int
        Sample rate in Hz.

    num_channels : int
        Number of audio channels.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    waveform : np.ndarray
        The generated dummy waveform.

    sample_rate : int
        The sample rate of the waveform.

    """
    np.random.seed(seed)
    num_samples = int(duration_sec * sample_rate)
    waveform = 0.1 * np.random.randn(num_channels, num_samples)
    if num_channels == 1:
        waveform = waveform.squeeze(0)
    return waveform, sample_rate
