# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the Whisper model."""

from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import silence_transformers
from ..utils.validation import validate_enum
from .base import FrameLevelFeatureModel


class WhisperVariant(str, Enum):
    """Enumeration of supported Whisper model variants."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"

    @property
    def model_name(self) -> str:
        """Return the model name corresponding to the variant.

        Returns
        -------
        out : str
            The model name corresponding to the variant.

        """
        return f"openai/whisper-{self.value}"


class WhisperModel(FrameLevelFeatureModel):
    """A class for the Whisper model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the Whisper model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, WhisperVariant, WhisperVariant.SMALL)
        self._model_id = f"whisper-{self.variant.value}"

        self.processor = None
        self.model = None

    def load(self, model_dir: str, quiet: bool = False) -> None:
        """Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model files will be stored.

        quiet : bool, optional
            Whether to suppress output during the loading process.

        """
        if self.processor is not None and self.model is not None:
            return

        from transformers import WhisperModel, WhisperProcessor

        with silence_transformers(quiet):
            self.processor = WhisperProcessor.from_pretrained(
                self.variant.model_name, cache_dir=model_dir
            )
            self.model = WhisperModel.from_pretrained(
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
        if self.processor is None or self.model is None:
            raise RuntimeError("Model not loaded. Call 'load' method first.")

        inputs = self.processor(
            audio.array,
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=audio.sample_rate,
        )

        with torch.inference_mode():
            input_features = self.model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(device=self.device, dtype=self.model.encoder.dtype)

            hidden_states = self.model.encoder(
                input_features,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states

            vectors = torch.concat([hidden_states[i] for i in layers], dim=-1)

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
            WhisperVariant.TINY: 4,
            WhisperVariant.BASE: 6,
            WhisperVariant.SMALL: 12,
            WhisperVariant.MEDIUM: 24,
            WhisperVariant.LARGE: 32,
            WhisperVariant.LARGE_V2: 32,
            WhisperVariant.LARGE_V3: 32,
        }
        return variant_map.get(self.variant, 0)

    @property
    def center_offset(self) -> int:
        """Get the center offset of the model.

        Returns
        -------
        out : int
            The center offset in samples.

        """
        return 0

    @property
    def chunk_length_sec(self) -> int | None:
        """Get the chunk length in seconds of the model.

        Returns
        -------
        out : int | None
            The chunk length in seconds, or None if not applicable.

        """
        return 30
