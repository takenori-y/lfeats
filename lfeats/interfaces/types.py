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

    def fit_to_length(self, length: int) -> Features:
        """Fit the features to the specified length by padding or cutting.

        Parameters
        ----------
        length : int
            The desired number of frames.

        Returns
        -------
        out : Features
            A new Features instance fitted to the specified length.

        """
        diff = length - self.length
        if diff == 0:
            return self

        if diff > 0:
            padding = (0, diff)
            if isinstance(self.data, np.ndarray):
                fitted_vectors = np.pad(self.data, ((0, 0), padding, (0, 0)))
            else:
                fitted_vectors = F.pad(self.data, (0, 0, *padding))
        else:
            fitted_vectors = self.data[:, :length]
        return Features(data=fitted_vectors, source=self.source)
