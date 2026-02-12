# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for extracting features from a specified model."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from platformdirs import user_cache_dir

from ..models import MODEL_MAP, Audio, Features


class Extractor:
    """A class for extracting features from a specified model."""

    def __init__(
        self,
        model_family: str,
        variant: str | None = None,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the Extractor with the specified model.

        Parameters
        ----------
        model_family : str
            The name of the model to use.

        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        cache_dir : str | None, optional
            The directory to cache the model files.

        """
        cls = MODEL_MAP[model_family]
        self.model = cls(variant=variant, device=device)

        if cache_dir is None:
            self.cache_dir = user_cache_dir("lfeats")
        else:
            self.cache_dir = cache_dir

    def load(self) -> None:
        """Load the model from the cache directory."""
        self.model.load(self.cache_dir)

    def __call__(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int,
        layers: int | Sequence[int] | Literal["all", "last"] = "last",
    ) -> Features:
        """Extract features from the input waveform.

        Parameters
        ----------
        waveform : np.ndarray | torch.Tensor
            The input waveform data with shape (T,) or (B, T).

        sample_rate : int
            The sample rate of the input waveform.

        layers : int | list[int] | Literal["all", "last"], optional
            The layer(s) from which to extract features.

        Returns
        -------
        out : Features
            The extracted features.

        """
        self.load()

        expected_num_frames = int(np.ceil(waveform.shape[-1] / self.model.frame_shift))

        if self.model.center_offset > 0:
            padding = (self.model.center_offset, self.model.center_offset - 1)
            if isinstance(waveform, np.ndarray):
                waveform = np.pad(waveform, padding)
            else:
                waveform = F.pad(waveform, padding)

        audio = Audio(waveform, sample_rate)
        features = self.model.extract_features(audio, self._normalize_layers(layers))
        features.vectors = features.vectors[:, :expected_num_frames]
        return features

    def _normalize_layers(
        self, layers: int | Sequence[int] | Literal["all", "last"]
    ) -> list[int]:
        """Normalize the layer specification to a list of layer indices.

        Parameters
        ----------
        layers : int | list[int] | Literal["all", "last"]
            The layer(s) from which to extract features.

        Returns
        -------
        out : list[int]
            The normalized list of layer indices.

        Raises
        ------
        ValueError
            If the layers specification is invalid.

        """
        if layers == "all":
            return list(range(self.model.num_layers))
        if layers == "last":
            return [self.model.num_layers - 1]
        if isinstance(layers, str):
            raise ValueError(f"Invalid layers specification string: {layers}")
        if isinstance(layers, int):
            layers = [layers]
        if isinstance(layers, Sequence):
            layers = [i if i >= 0 else self.model.num_layers + i for i in layers]
            if any(i < 0 or i >= self.model.num_layers for i in layers):
                raise ValueError(f"Layer index out of range: {layers}")
            return layers
        raise ValueError(f"Invalid layers specification type: {layers}")
