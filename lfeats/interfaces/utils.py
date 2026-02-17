# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Utility functions for interfaces."""

import numpy as np
import torch

from .types import Audio


def create_audio_object(
    source: np.ndarray | torch.Tensor | Audio, sample_rate: int | None
) -> Audio:
    """Create an Audio object from the input source.

    Parameters
    ----------
    source : np.ndarray | torch.Tensor | Audio
        The input waveform data with shape (T,) or (B, T) or an Audio object.

    sample_rate : int | None
        The sample rate of the input waveform.

    Returns
    -------
    out : Audio
        The created Audio object.

    Raises
    ------
    ValueError
        If the sample_rate is invalid for the given source.

    """
    if isinstance(source, Audio):
        if sample_rate is not None and source.sample_rate != sample_rate:
            raise ValueError(
                "sample_rate must be None when source is an Audio object "
                "or must match the sample rate of the Audio object."
            )
        audio = source
    else:
        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided when source is not an Audio object."
            )
        audio = Audio(source, sample_rate)
    return audio
