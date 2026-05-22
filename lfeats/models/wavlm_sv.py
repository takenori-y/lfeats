# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the WavLM model."""

from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import silence_transformers
from ..utils.paths import sanitize
from ..utils.validation import validate_enum
from .base import UtteranceLevelFeatureModel


class WavLMSVVariant(str, Enum):
    """Enumeration of supported WavLM-for-speaker-verification model variants."""

    BASE = "base"
    BASE_PLUS = "base+"

    @property
    def model_name(self) -> str:
        """Return the model name corresponding to the variant.

        Returns
        -------
        out : str
            The model name corresponding to the variant.

        """
        return f"microsoft/wavlm-{sanitize(self.value)}-sv"


class WavLMSVModel(UtteranceLevelFeatureModel):
    """A class for the WavLM-for-speaker-verification model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the WavLM-for-speaker-verification model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, WavLMSVVariant, WavLMSVVariant.BASE_PLUS)
        self._model_id = f"wavlm-{self.variant.value}"

        self.feature_extractor = None

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

        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        with silence_transformers(quiet):
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.variant.model_name, cache_dir=model_dir
            )
            self.model = WavLMForXVector.from_pretrained(
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
        if self.feature_extractor is None or self.model is None:
            raise RuntimeError("Model not loaded. Call 'load' method first.")

        with torch.inference_mode():
            inputs = self.feature_extractor(
                [x for x in audio.array],
                sampling_rate=audio.sample_rate,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            embeddings = self.model(**inputs).embeddings
            embeddings = embeddings.unsqueeze(1)

        return Features(data=embeddings, source=self.model_id)
