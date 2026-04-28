# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A resampler implementation using scipy."""

from enum import Enum
from typing import cast

import numpy as np

from ..interfaces.types import Audio
from ..utils.validation import validate_enum
from .base import BaseResampler


class ScipyPreset(str, Enum):
    """Presets for the ScipyResampler."""

    FFT = "fft"
    POLYPHASE = "poly"


class ScipyResampler(BaseResampler):
    """A class for resampling audio using scipy."""

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        preset: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the ScipyResampler.

        Parameters
        ----------
        src_rate : int
            The source sample rate in Hz.

        dst_rate : int
            The destination sample rate in Hz.

        preset : str | None, optional
            The preset for the resampler.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(src_rate, dst_rate, preset, device)

        self.preset = validate_enum(preset, ScipyPreset, ScipyPreset.POLYPHASE)

        from scipy.signal import resample, resample_poly

        if self.preset == ScipyPreset.FFT:

            def _resample_fft(x):
                n_samples = int(np.ceil(x.shape[-1] * dst_rate / src_rate))
                return resample(x, n_samples, axis=-1)

            self.resampler = _resample_fft
        elif self.preset == ScipyPreset.POLYPHASE:

            def _resample_poly(x):
                divisor = np.gcd(src_rate, dst_rate)
                return resample_poly(
                    x, dst_rate // divisor, src_rate // divisor, axis=-1
                )

            self.resampler = _resample_poly
        else:
            raise ValueError(f"Unsupported preset: {self.preset}")

    def resample_impl(self, audio: Audio) -> Audio:
        """Resample the given audio to the target sample rate.

        Parameters
        ----------
        audio : Audio
            The input audio to be resampled.

        Returns
        -------
        out : Audio
            The resampled audio.

        """
        samples = cast(np.ndarray, self.resampler(audio.array))
        return Audio(data=samples, sample_rate=self.dst_rate)
