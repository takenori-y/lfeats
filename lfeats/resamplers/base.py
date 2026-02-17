# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module defining the base resampler class for audio processing."""

from abc import ABC, abstractmethod

from ..models.base import Audio


class BaseResampler(ABC):
    """An abstract base class for audio resamplers."""

    def __init__(
        self, src_rate: int, dst_rate: int, preset: str | None, device: str
    ) -> None:
        """Initialize the BaseResampler with source and destination sample rates.

        Parameters
        ----------
        src_rate : int
            The source sample rate in Hz.

        dst_rate : int
            The destination sample rate in Hz.

        preset : str | None
            The preset for the resampler.

        device : str
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.device = device

    def resample(self, audio: Audio) -> Audio:
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
        if audio.sample_rate != self.src_rate:
            raise ValueError(
                f"Input sample rate {audio.sample_rate} does not match "
                f"resampler source sample rate {self.src_rate}."
            )
        if audio.sample_rate == self.dst_rate:
            return audio
        return self.resample_impl(audio)

    @abstractmethod
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
        raise NotImplementedError
