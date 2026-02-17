# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A resampler implemented using the soxr library."""

from enum import Enum

from ..interfaces.types import Audio
from ..utils.validation import validate_enum
from .base import BaseResampler


class SoxrPreset(str, Enum):
    """Presets for the SoxrResampler."""

    QQ = "quick"
    LQ = "low"
    MQ = "medium"
    HQ = "high"
    VHQ = "very-high"

    @property
    def quality(self) -> str:
        """Return the quality string for the soxr resampler."""
        preset_map = {
            SoxrPreset.QQ: "QQ",
            SoxrPreset.LQ: "LQ",
            SoxrPreset.MQ: "MQ",
            SoxrPreset.HQ: "HQ",
            SoxrPreset.VHQ: "VHQ",
        }
        return preset_map[self]


class SoxrResampler(BaseResampler):
    """A class for resampling audio using the soxr library."""

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

        self.preset = validate_enum(preset, SoxrPreset, SoxrPreset.HQ)

        import soxr

        self.resampler = soxr.resample

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
        samples = self.resampler(
            audio.array.T,
            self.src_rate,
            self.dst_rate,
            quality=self.preset.quality,
        ).T
        return Audio(data=samples, sample_rate=self.dst_rate)
