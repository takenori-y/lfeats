# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for managing multiple resamplers."""

from .base import BaseResampler


class ResamplerManager:
    """A manager for multiple resamplers."""

    def __init__(
        self, resampler_cls: type[BaseResampler], preset: str | None, device: str
    ) -> None:
        """Initialize the ResamplerManager with the specified resampler class.

        Parameters
        ----------
        resampler_cls : type[BaseResampler]
            The resampler class to manage.

        preset : str
            The preset for the resampler.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        self.resampler_cls = resampler_cls
        self.preset = preset
        self.device = device

        self._cache: dict[tuple[int, int], BaseResampler] = {}

    def get_resampler(self, src_rate: int, dst_rate: int) -> BaseResampler:
        """Get the resampler instance.

        Parameters
        ----------
        src_rate : int
            The source sample rate.

        dst_rate : int
            The destination sample rate.

        Returns
        -------
        out : BaseResampler
            The resampler instance.

        """
        key = (src_rate, dst_rate)
        if key not in self._cache:
            self._cache[key] = self.resampler_cls(
                src_rate=src_rate,
                dst_rate=dst_rate,
                preset=self.preset,
                device=self.device,
            )
        return self._cache[key]
