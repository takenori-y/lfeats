# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A resampler implementation using torchaudio."""

from enum import Enum

import torch

from ..interfaces.types import Audio
from .base import BaseResampler


class TorchAudioPreset(str, Enum):
    """Presets for the TorchAudioResampler."""

    KAISER_FAST = "kaiser-fast"
    KAISER_BEST = "kaiser-best"


class TorchAudioResampler(BaseResampler):
    """A class for resampling audio using torchaudio."""

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        preset: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the TorchAudioResampler.

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

        if preset is None:
            preset = TorchAudioPreset.KAISER_BEST
        try:
            preset = TorchAudioPreset(preset)
        except ValueError as e:
            raise ValueError(
                f"Unsupported preset '{preset}'. "
                f"Supported presets are: {[v.value for v in TorchAudioPreset]}"
            ) from e

        if preset == TorchAudioPreset.KAISER_FAST:
            params = {
                "resampling_method": "sinc_interp_kaiser",
                "lowpass_filter_width": 16,
                "rolloff": 0.85,
                "beta": 8.555504641634386,
            }
        elif preset == TorchAudioPreset.KAISER_BEST:
            params = {
                "resampling_method": "sinc_interp_kaiser",
                "lowpass_filter_width": 64,
                "rolloff": 0.9475937167399596,
                "beta": 14.769656459379492,
            }

        import torchaudio

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=src_rate, new_freq=dst_rate, dtype=torch.float32, **params
        ).to(device)

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
        samples = self.resampler(audio.tensor.to(self.device))
        return Audio(data=samples, sample_rate=self.dst_rate)
