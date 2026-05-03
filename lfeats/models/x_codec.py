# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the Higgs Audio tokenizer."""

from enum import Enum
from typing import Any

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import silence_transformers
from ..utils.validation import validate_enum
from .base import TokenLevelFeatureModel


class XCodecVariant(str, Enum):
    """Enumeration of supported X-Codec variants."""

    HUBERT = "hubert"
    WAVLM = "wavlm"

    @property
    def model_name(self) -> str:
        """Return the model name corresponding to the variant.

        Returns
        -------
        out : str
            The model name corresponding to the variant.

        """
        base = f"hf-audio/xcodec-{self.value}"
        if self.value == "hubert":
            return f"{base}-librispeech"
        elif self.value == "wavlm":
            return f"{base}-more-data"
        return base


class XCodecModel(TokenLevelFeatureModel):
    """A class for the X-Codec model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the X-Codec model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(
            variant, XCodecVariant, XCodecVariant.HUBERT
        )
        self._model_id = f"x-codec-{self.variant.value}"

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

        from transformers import AutoFeatureExtractor, XcodecModel

        with silence_transformers(quiet):
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.variant.model_name, cache_dir=model_dir
            )
            self.model = XcodecModel.from_pretrained(
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
                raw_audio=[x for x in audio.array],
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt",
            ).to(self.device)

            encoder_outputs: Any = self.model.encode(inputs["input_values"])
            indices = encoder_outputs.audio_codes  # (B, Q, N)
            indices = indices.transpose(0, 1)
            vectors = self.model.quantizer.decode(indices)  # (B, D, N)
            vectors = vectors.transpose(1, 2)

        return Features(data=vectors, source=self.model_id)
