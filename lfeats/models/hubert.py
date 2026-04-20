# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the HuBERT model."""

from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import silence_transformers
from ..utils.validation import validate_enum
from .base import FrameLevelFeatureModel


class HuBERTVariant(str, Enum):
    """Enumeration of supported HuBERT model variants."""

    BASE = "base"
    LARGE = "large"
    XLARGE = "xlarge"

    @property
    def model_name(self) -> str:
        """Return the model name corresponding to the variant.

        Returns
        -------
        out : str
            The model name corresponding to the variant.

        """
        base = f"facebook/hubert-{self.value}"
        if self.value == "base":
            return f"{base}-ls960"
        elif self.value == "large":
            return f"{base}-ll60k"
        elif self.value == "xlarge":
            return f"{base}-ll60k"
        return base


class HuBERTModel(FrameLevelFeatureModel):
    """A class for the HuBERT model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the HuBERT model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, HuBERTVariant, HuBERTVariant.BASE)
        self._model_id = f"hubert-{self.variant.value}"

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

        from transformers import HubertModel

        with silence_transformers(quiet):
            self.model = HubertModel.from_pretrained(
                self.variant.model_name, cache_dir=model_dir
            )
            self.model.eval()
            self.model.to(self.device)  # type: ignore

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
            raise RuntimeError("Model not loaded. Call 'load' method first.")

        if self.variant != HuBERTVariant.BASE:
            audio = audio.normalize()

        with torch.inference_mode():
            hidden_states = self.model(
                input_values=audio.tensor.to(self.device), output_hidden_states=True
            ).hidden_states

            vectors = torch.cat([hidden_states[i] for i in layers], dim=-1)

        return Features(data=vectors, source=self.model_id, layers=layers)

    @property
    def num_layers(self) -> int:
        """Get the number of layers in the model.

        Returns
        -------
        out : int
            The number of layers.

        """
        variant_map = {
            HuBERTVariant.BASE: 12,
            HuBERTVariant.LARGE: 24,
            HuBERTVariant.XLARGE: 48,
        }
        return variant_map.get(self.variant, 0)
