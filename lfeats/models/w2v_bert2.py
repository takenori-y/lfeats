# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the w2v-BERT 2.0 model."""

from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import setup_transformers
from ..utils.validation import validate_enum, validate_length
from .base import FrameLevelFeatureModel


class W2VBert2Variant(str, Enum):
    """Enumeration of supported w2v-BERT 2.0 model variants."""

    BASE = "base"

    @property
    def model_name(self) -> str:
        """Return the model name corresponding to the variant.

        Returns
        -------
        out : str
            The model name corresponding to the variant.

        """
        return "facebook/w2v-bert-2.0"


class W2VBert2Model(FrameLevelFeatureModel):
    """A class for the w2v-BERT 2.0 model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the w2v-BERT 2.0 model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, W2VBert2Variant, W2VBert2Variant.BASE)
        self._model_id = f"w2v-bert2-{self.variant.value}"

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

        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

        with setup_transformers(quiet):
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.variant.model_name, cache_dir=model_dir
            )
            self.model = Wav2Vec2BertModel.from_pretrained(
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
            wavs = validate_length(audio.tensor, 560)
            inputs = self.feature_extractor(
                raw_speech=[x.numpy() for x in wavs],
                sampling_rate=self.feature_extractor.sampling_rate,
                pad_to_multiple_of=None,
                return_tensors="pt",
            ).to(self.device)

            hidden_states = self.model(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
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
        return 24

    @property
    def center_offset(self) -> int:
        """Get the center offset of the model.

        Two consecutive filterbank frames, each with a window of 25 ms and a shift of
        10 ms, are stacked into one frame, resulting in a center offset of 17.5 ms.

        Returns
        -------
        out : int
            The center offset in samples.

        """
        return int(17.5 * self.sample_rate / 1000)
