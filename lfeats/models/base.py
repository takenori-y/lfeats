# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module defining the base model and data structures for feature extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class Backend(str, Enum):
    """Enumeration of inference backends."""

    TORCH = "torch"


@dataclass
class Audio:
    """Dataclass to hold input audio data."""

    samples: np.ndarray | torch.Tensor  # The input waveform data with shape (B, T).
    sample_rate: int  # The sample rate of the input waveform.

    def __post_init__(self):
        """Validate the input data."""
        if self.ndim == 1:
            self.samples = self.samples[None, :]
        if self.ndim != 2:
            raise ValueError("samples must be 1D or 2D array/tensor.")

        if isinstance(self.samples, np.ndarray):
            if self.samples.dtype != np.float32:
                self.samples = self.samples.astype(np.float32)
        elif isinstance(self.samples, torch.Tensor):
            if self.samples.dtype != torch.float32:
                self.samples = self.samples.to(torch.float32)
        else:
            raise TypeError("samples must be a NumPy array or a PyTorch tensor.")

        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")

    def array(self) -> np.ndarray:
        """Get the audio samples as a NumPy array.

        Returns
        -------
        out : np.ndarray
            The audio samples as a NumPy array.

        """
        if isinstance(self.samples, np.ndarray):
            return self.samples
        return self.samples.detach().cpu().numpy()

    def tensor(self) -> torch.Tensor:
        """Get the audio samples as a PyTorch tensor.

        Returns
        -------
        out : torch.Tensor
            The audio samples as a PyTorch tensor.

        """
        if isinstance(self.samples, torch.Tensor):
            return self.samples
        return torch.from_numpy(self.samples)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the audio samples.

        Returns
        -------
        out : int
            The number of dimensions.

        """
        if isinstance(self.samples, np.ndarray):
            return self.samples.ndim
        return self.samples.dim()


@dataclass
class Features:
    """Dataclass to hold extracted features."""

    vectors: np.ndarray | torch.Tensor  # The extracted features with shape (B, N, D).

    @property
    def array(self) -> np.ndarray:
        """Get the features as a NumPy array.

        Returns
        -------
        out : np.ndarray
            The extracted features as a NumPy array.

        """
        if isinstance(self.vectors, np.ndarray):
            return self.vectors
        return self.vectors.detach().cpu().numpy()

    @property
    def tensor(self) -> torch.Tensor:
        """Get the features as a PyTorch tensor.

        Returns
        -------
        out : torch.Tensor
            The extracted features as a PyTorch tensor.

        """
        if isinstance(self.vectors, torch.Tensor):
            return self.vectors
        return torch.from_numpy(self.vectors)


class BaseModel(ABC):
    """An abstract base class for feature extraction models."""

    def extract_features(self, audio: Audio, layers: list[int]) -> Features:
        """Extract features from the input audio data.

        Parameters
        ----------
        audio : Audio
            The input audio data.

        layers : list[int]
            The layer(s) from which to extract features.

        Returns
        -------
        out : Features
            The extracted features.

        Raises
        ------
        ValueError
            If the input sample rate does not match the model sample rate.

        """
        if audio.sample_rate != self.sample_rate:
            raise ValueError(
                f"Input sample rate {audio.sample_rate} does not match "
                f"model sample rate {self.sample_rate}."
            )
        return self.extract_features_impl(audio, layers)

    @abstractmethod
    def extract_features_impl(self, audio: Audio, layers: list[int]) -> Features:
        """Extract features from the input audio data.

        Parameters
        ----------
        audio : Audio
            The input audio data.

        layers : list[int]
            The layer(s) from which to extract features.

        Returns
        -------
        out : Features
            The extracted features.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get the sample rate required by the model.

        Returns
        -------
        out : int
            The sample rate in Hz.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Get the number of available layers in the model.

        Returns
        -------
        out : int
            The number of layers.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_shift(self) -> int:
        """Get the frame shift of the model.

        Returns
        -------
        out : int
            The frame shift in samples.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def center_offset(self) -> int:
        """Get the center offset of the model.

        Returns
        -------
        out : int
            The center offset in samples.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """Get the backend framework used by the model.

        Returns
        -------
        out : Backend
            The backend framework.

        """
        raise NotImplementedError
