# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the ECAPA-TDNN model."""

from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import silence_hf_hub
from ..utils.paths import setup_third_party_path
from ..utils.validation import validate_enum
from .base import UtteranceLevelFeatureModel


class EcapaTDNNVariant(str, Enum):
    """Enumeration of supported ECAPA-TDNN model variants."""

    BASE = "base"


class EcapaTDNNModel(UtteranceLevelFeatureModel):
    """A class for the ECAPA-TDNN model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the ECAPA-TDNN model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, EcapaTDNNVariant, EcapaTDNNVariant.BASE)
        self._model_id = f"ecapa-tdnn-{self.variant.value}"

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

        setup_third_party_path()

        from lfeats.third_party.speechbrain.inference.classifiers import (
            EncoderClassifier,
        )

        with silence_hf_hub(quiet):
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )
            if self.model is None:
                raise RuntimeError("Failed to load the model.")
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
            vectors = self.model.encode_batch(audio.tensor.to(self.device))  # (B, N, D)

        return Features(data=vectors, source=self.model_id)
