# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for resampling audio using a specified resampler."""

import numpy as np
import torch

from ..resamplers import RESAMPLER_MAP, ResamplerManager
from .types import Audio
from .utils import create_audio_object


class Resampler:
    """A class for resampling audio using a specified resampler."""

    def __init__(
        self,
        resampler_type: str = "torchaudio",
        resampler_preset: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the Resampler with the specified resampler type.

        Parameters
        ----------
        resampler_type : str, optional
            The type of resampler to use.

        resampler_preset : str | None, optional
            The preset for the resampler.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        try:
            cls = RESAMPLER_MAP[resampler_type]
            self.resampler_manager = ResamplerManager(
                cls, preset=resampler_preset, device=device
            )
        except KeyError as e:
            raise ValueError(
                f"Unsupported resampler '{resampler_type}'. "
                f"Supported resamplers are: {[k for k in RESAMPLER_MAP.keys()]}"
            ) from e

    def to(self, device: str) -> None:
        """Move the resampler to the specified device.

        Parameters
        ----------
        device : str
            The device to move the resampler to (e.g., 'cpu' or 'cuda').

        """
        self.resampler_manager.to(device)

    def __call__(
        self,
        source: np.ndarray | torch.Tensor | Audio,
        *,
        src_rate: int | None = None,
        dst_rate: int = 16000,
    ) -> Audio:
        """Resample the given audio to the target sample rate.

        Parameters
        ----------
        source : np.ndarray | torch.Tensor | Audio
            The input waveform data with shape (T,) or (B, T) or an Audio object.

        src_rate : int | None, optional
            The source sample rate in Hz.

        dst_rate : int, optional
            The destination sample rate in Hz.

        Returns
        -------
        out : Audio
            The resampled audio with shape (B, T').

        Examples
        --------
        >>> import lfeats
        >>> import numpy as np
        >>>
        >>> sample_rate = 16000
        >>> waveform = np.random.randn(sample_rate)
        >>>
        >>> resampler = lfeats.Resampler(resampler_type="torchaudio")
        >>> resampled_audio = resampler(waveform, src_rate=sample_rate, dst_rate=8000)
        >>> resampled_audio.shape
        (1, 8000)

        """
        audio = create_audio_object(source, src_rate)
        if audio.sample_rate != dst_rate:
            resampler = self.resampler_manager.get_resampler(
                audio.sample_rate, dst_rate
            )
            audio = resampler.resample(audio)
        return audio
