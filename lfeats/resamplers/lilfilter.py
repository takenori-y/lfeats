# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A resampler implementation using lilfilter."""

import torch

from ..interfaces.types import Audio
from .base import BaseResampler


class LilFilterResampler(BaseResampler):
    """A class for resampling audio using lilfilter."""

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        preset: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the LilfilterResampler.

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

        self.device = device

        import lilfilter

        self.resampler = lilfilter.Resampler(
            src_rate,
            dst_rate,
            dtype=torch.float32,
        )
        self.resampler.weights = self.resampler.weights.to(self.device)

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
        samples = self.resampler.resample(audio.tensor.to(self.device))
        return Audio(data=samples, sample_rate=self.dst_rate)
