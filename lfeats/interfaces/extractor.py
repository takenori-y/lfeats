# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for extracting features from a specified model."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from platformdirs import user_cache_dir

from ..models import MODEL_MAP, ModelManager
from ..resamplers import RESAMPLER_MAP, ResamplerManager
from .types import Audio, Features


class Extractor:
    """A class for extracting features from a specified model."""

    def __init__(
        self,
        model_name: str,
        model_variant: str | None = None,
        resampler_type: str = "torchaudio",
        resampler_preset: str | None = None,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the Extractor with the specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to use.

        model_variant : str | None, optional
            The variant of the model to use.

        resampler_type : str, optional
            The type of resampler to use.

        resampler_preset : str | None, optional
            The preset for the resampler. If None, the highest quality preset is used.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        cache_dir : str | None, optional
            The directory to cache the model files.

        """
        try:
            cls = MODEL_MAP[model_name]
            self.model_manager = ModelManager(cls, variant=model_variant, device=device)
        except KeyError as e:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models are: {[k for k in MODEL_MAP.keys()]}"
            ) from e

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

        if cache_dir is None:
            self.cache_dir = user_cache_dir("lfeats")
        else:
            self.cache_dir = cache_dir

    def __call__(
        self,
        source: np.ndarray | torch.Tensor | Audio,
        sample_rate: int | None = None,
        layers: int | Sequence[int] | Literal["all", "last"] = "last",
    ) -> Features:
        """Extract features from the input waveform.

        Parameters
        ----------
        source : np.ndarray | torch.Tensor | Audio
            The input waveform data with shape (T,) or (B, T) or an Audio object.

        sample_rate : int | None, optional
            The sample rate of the input waveform.
            Must be provided if `source` is not an Audio object.

        layers : int | list[int] | Literal["all", "last"], optional
            The layer(s) from which to extract features.

        Returns
        -------
        out : Features
            The extracted features.

        Raises
        ------
        ValueError
            If sample_rate is not provided when source is not an Audio object.

        """
        model = self.model_manager.get_model()
        model.load(self.cache_dir)

        if isinstance(source, Audio):
            audio = source
        else:
            if sample_rate is None:
                raise ValueError(
                    "sample_rate must be provided when source is not an Audio object."
                )
            audio = Audio(source, sample_rate)
        expected_num_frames = int(np.ceil(audio.length / model.frame_shift))

        if sample_rate != model.sample_rate:
            resampler = self.resampler_manager.get_resampler(
                audio.sample_rate, model.sample_rate
            )
            audio = resampler.resample(audio)

        if model.center_offset > 0:
            padding = (model.center_offset, model.center_offset - 1)
            audio = audio.pad(padding)

        # Extract features.
        features = model.extract_features(audio, self._normalize_layers(layers))
        actual_num_frames = features.length

        if expected_num_frames != actual_num_frames:
            features = features.fit_to_length(expected_num_frames)

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
        num_layers = self.model_manager.get_model().num_layers

        if layers == "all":
            return list(range(num_layers))
        if layers == "last":
            return [num_layers - 1]
        if isinstance(layers, str):
            raise ValueError(f"Invalid layers specification string: {layers}")

        if isinstance(layers, int):
            layers = [layers]
        if isinstance(layers, Sequence):
            layers = [i if i >= 0 else num_layers + i for i in layers]
            if any(i < 0 or i >= num_layers for i in layers):
                raise ValueError(f"Layer index out of range: {layers}")
            return layers

        raise ValueError(f"Invalid layers specification type: {layers}")
