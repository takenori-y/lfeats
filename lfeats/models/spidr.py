# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the SpidR model."""

from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import safe_torch_hub_load
from ..utils.validation import validate_enum
from .base import FrameLevelFeatureModel


class SpidRVariant(str, Enum):
    """Enumeration of supported SpindR model variants."""

    BASE = "base"


class SpidRModel(FrameLevelFeatureModel):
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
        self._model_id = f"spidr-{self.variant.value}"

        self.model = None

    def load(self, model_dir: str, quiet: bool = False) -> None:
        """Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model checkpoint will be stored.

        quiet : bool, optional
            Whether to suppress output during the loading process.

        """
        if self.model is not None:
            return

        self.model = safe_torch_hub_load(
            "facebookresearch/spidr", "spidr_base", model_dir, quiet=quiet
        )
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

        # The model expects standardized audio.
        audio = audio.normalize()

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
