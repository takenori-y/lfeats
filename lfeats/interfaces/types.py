# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module defining data types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F


class Backend(str, Enum):
    """Enumeration of running backends."""

    TORCH = "torch"


@dataclass
class Container:
    """Dataclass to hold generic data."""

    data: np.ndarray | torch.Tensor  # The input data.

    def __post_init__(self):
        """Validate the input data."""
        if isinstance(self.data, np.ndarray):
            if self.data.dtype != np.float32:
                self.data = self.data.astype(np.float32)
        elif isinstance(self.data, torch.Tensor):
            if self.data.dtype != torch.float32:
                self.data = self.data.to(torch.float32)
        else:
            raise TypeError("data must be a NumPy array or a PyTorch tensor.")

    def is_array(self) -> bool:
        """Check if the data is a NumPy array.

        Returns
        -------
        out : bool
            True if the data is a NumPy array, False otherwise.

        """
        return isinstance(self.data, np.ndarray)

    def is_tensor(self) -> bool:
        """Check if the data is a PyTorch tensor.

        Returns
        -------
        out : bool
            True if the data is a PyTorch tensor, False otherwise.

        """
        return isinstance(self.data, torch.Tensor)

    @property
    def array(self) -> np.ndarray:
        """Get the data as a NumPy array.

        Returns
        -------
        out : np.ndarray
            The data as a NumPy array.

        """
        if isinstance(self.data, np.ndarray):
            return self.data
        return self.data.detach().cpu().numpy()

    @property
    def tensor(self) -> torch.Tensor:
        """Get the data as a PyTorch tensor.

        Returns
        -------
        out : torch.Tensor
            The data as a PyTorch tensor.

        """
        if isinstance(self.data, torch.Tensor):
            return self.data
        return torch.from_numpy(self.data)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the data.

        Returns
        -------
        out : int
            The number of dimensions of the data.

        """
        if isinstance(self.data, np.ndarray):
            return self.data.ndim
        return self.data.dim()

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the data.

        Returns
        -------
        out : tuple[int, ...]
            The shape of the data.

        """
        return self.data.shape

    def zeros(self, shape: tuple[int, ...]) -> np.ndarray | torch.Tensor:
        """Create a new array/tensor of zeros with the same type as the data.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the new array/tensor.

        Returns
        -------
        out : np.ndarray | torch.Tensor
            A new array/tensor of zeros with the specified shape.

        """
        if isinstance(self.data, np.ndarray):
            return np.zeros(shape, dtype=np.float32)
        return torch.zeros(shape, dtype=torch.float32, device=self.data.device)


@dataclass
class Audio(Container):
    """Dataclass to hold audio data."""

    sample_rate: int  # The sample rate of the input waveform.

    def __post_init__(self):
        """Validate the input data."""
        super().__post_init__()

        if self.ndim == 1:
            self.data = self.data[None, ...]
        if self.ndim != 2:
            raise ValueError("data must be 1D or 2D array/tensor.")

        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")

    @property
    def length(self) -> int:
        """Get the length of the audio samples.

        Returns
        -------
        out : int
            The length of the audio samples.

        """
        return self.data.shape[1]

    def pad(self, padding: tuple[int, int]) -> Audio:
        """Pad the audio samples.

        Parameters
        ----------
        padding : tuple[int, int]
            The amount of padding to add at the beginning and end of the samples.

        Returns
        -------
        out : Audio
            A new Audio instance with padded samples.

        """
        if isinstance(self.data, np.ndarray):
            padded_samples = np.pad(self.data, ((0, 0), padding))
        else:
            padded_samples = F.pad(self.data, padding)
        return Audio(data=padded_samples, sample_rate=self.sample_rate)


@dataclass
class Features(Container):
    """Dataclass to hold latent features."""

    source: str  # The source of the features.

    def __post_init__(self):
        """Validate the input data."""
        super().__post_init__()

        if self.ndim == 2:
            self.data = self.data[None, ...]
        if self.ndim != 3:
            raise ValueError("data must be 2D or 3D array/tensor.")

    @property
    def length(self) -> int:
        """Get the length of the audio samples.

        Returns
        -------
        out : int
            The length of the audio samples.

        """
        return self.data.shape[1]

    def trim(self, start: int, end: int) -> Features:
        """Trim the features along the time dimension.

        Parameters
        ----------
        start : int
            The starting frame index to trim from.

        end : int
            The ending frame index to trim to (exclusive).

        Returns
        -------
        out : Features
            A new Features instance with trimmed data.

        Raises
        ------
        ValueError
            If the start or end indices are out of bounds.

        """
        if start < 0 or end > self.length or start >= end:
            raise ValueError(
                f"Invalid parameters: start={start}, end={end}, length={self.length}"
            )

        if isinstance(self.data, np.ndarray):
            trimmed_data = self.array[:, start:end]
        else:
            trimmed_data = self.tensor[:, start:end]
        return Features(data=trimmed_data, source=self.source)

    def merge(self, other: Features, overlap_length: int = 0) -> Features:
        """Merge this Features instance with another one along the time dimension.

        Linearly crossfade the overlapping frames if overlap_length is greater than 0.

        Parameters
        ----------
        other : Features
            The other Features instance to merge with.

        overlap_length : int, optional
            The number of frames to overlap between the two features when merging.

        Returns
        -------
        out : Features
            A new Features instance with merged data.

        Raises
        ------
        ValueError
            If the sources are not the same or if the overlap_length is invalid.

        """
        if self.source != other.source:
            raise ValueError("Both Features instances must have the same source.")
        if overlap_length < 0 or overlap_length >= min(self.length, other.length):
            raise ValueError(
                f"Invalid parameters: overlap_length={overlap_length}, "
                f"length1={self.length}, length2={other.length}"
            )

        if overlap_length > 0:
            if isinstance(self.data, np.ndarray):
                fade_in = np.linspace(0, 1, overlap_length)[..., None]
                fade_out = 1 - fade_in
                merged_data = np.concatenate(
                    [
                        self.array[:, :-overlap_length],
                        (
                            self.array[:, -overlap_length:] * fade_out
                            + other.array[:, :overlap_length] * fade_in
                        ),
                        other.array[:, overlap_length:],
                    ],
                    axis=1,
                )
            else:
                fade_in = torch.linspace(
                    0, 1, overlap_length, device=self.tensor.device
                )[..., None]
                fade_out = 1 - fade_in
                merged_data = torch.cat(
                    [
                        self.tensor[:, :-overlap_length],
                        (
                            self.tensor[:, -overlap_length:] * fade_out
                            + other.tensor[:, :overlap_length] * fade_in
                        ),
                        other.tensor[:, overlap_length:],
                    ],
                    dim=1,
                )
        else:
            if isinstance(self.data, np.ndarray):
                merged_data = np.concatenate([self.array, other.array], axis=1)
            else:
                merged_data = torch.cat([self.tensor, other.tensor], dim=1)

        return Features(data=merged_data, source=self.source)
