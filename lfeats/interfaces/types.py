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

    def cut(self, span: tuple[int, int]) -> Features:
        """Cut the features to the specified span.

        Parameters
        ----------
        span : tuple[int, int]
            The start and end frame indices to cut.

        Returns
        -------
        out : Features
            A new Features instance with the specified span.

        """
        start, end = span
        if start < 0 or end > self.length or start >= end:
            raise ValueError("Invalid span values.")
        cut_vectors = self.data[:, start:end]
        return Features(data=cut_vectors, source=self.source)

    @staticmethod
    def concat(features_list: list[Features]) -> Features:
        """Concatenate a list of Features instances along the batch dimension.

        Parameters
        ----------
        features_list : list[Features]
            The list of Features instances to concatenate.

        Returns
        -------
        out : Features
            A new Features instance with concatenated data.

        Raises
        ------
        ValueError
            If the features_list is empty or if the sources are not the same.

        """
        if not features_list:
            raise ValueError("features_list must not be empty.")

        first_source = features_list[0].source
        for feat in features_list:
            if feat.source != first_source:
                raise ValueError("All Features instances must have the same source.")

        first_type = type(features_list[0].data)
        if first_type is np.ndarray:
            concatenated_data = np.concatenate(
                [feat.array for feat in features_list], axis=1
            )
        else:
            concatenated_data = torch.cat(
                [feat.tensor for feat in features_list], dim=1
            )

        return Features(data=concatenated_data, source=first_source)
