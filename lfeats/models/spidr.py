# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the SpidR model."""

from enum import Enum
from typing import Any

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import temporary_hub_dir
from ..utils.validation import validate_enum
from .base import BaseModel


class SpidRVariant(str, Enum):
    """Enumeration of supported SpindR model variants."""

    BASE = "base"


class SpidRModel(BaseModel):
    """A class for the SpidR model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the SpidR model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, SpidRVariant, SpidRVariant.BASE)

        self.model = None

    def load(self, model_dir: str) -> None:
        """Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model checkpoint will be stored.

        """
        if self.model is not None:
            return

        with temporary_hub_dir(model_dir):
            self.model: Any = torch.hub.load("facebookresearch/spidr", "spidr_base")
        self.model.eval()
        self.model.to(self.device)

    def extract_features_impl(self, audio: Audio, layers: list[int]) -> Features:
        """Extract features from the input audio using the model.

        Parameters
        ----------
        audio : Audio
            The input audio data with shape (B, T).

        layers : list[int]
            The layer(s) from which to extract features.

        Returns
        -------
        out : Features
            The extracted features.

        Raises
        ------
        RuntimeError
            If the model is not loaded.

        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call 'load' method first.")

        with torch.inference_mode():
            features = []

            x = self.model.feature_extractor(audio.tensor.to(self.device))
            x = self.model.feature_projection(x)
            features.append(x)

            x = x + self.model.student.pos_conv_embed(x)
            x = self.model.student.layer_norm(x)
            for layer in self.model.student.layers:
                x, layer_result = layer(x)
                features.append(layer_result)

            vectors = torch.concat([features[i] for i in layers], dim=-1)

        return Features(data=vectors, source=self.model_id, layers=layers)

    @property
    def num_layers(self) -> int:
        """Get the number of available layers in the model.

        Returns
        -------
        out : int
            The number of layers.

        """
        return 13

    @property
    def model_id(self) -> str:
        """Get the model identifier.

        Returns
        -------
        out : str
            The model identifier.

        """
        return f"spidr-{self.variant.value}"
