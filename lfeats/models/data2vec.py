# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the Whisper model."""

import os
import sys
from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import download_file
from ..utils.validation import validate_enum
from .base import BaseModel


class Data2VecVariant(str, Enum):
    """Enumeration of supported data2vec model variants."""

    BASE = "base"
    LARGE = "large"

    @property
    def checkpoint_filename(self) -> str:
        """Returns the corresponding checkpoint filename for the variant.

        Returns
        -------
        out : str
            The checkpoint filename.

        """
        if self.value == "base":
            return "audio_base_ls.pt"
        elif self.value == "large":
            return "vox_pretrained.pt"
        raise ValueError(f"Unsupported data2vec variant: {self.value}")


class Data2VecModel(BaseModel):
    """A class for the data2vec model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the data2vec model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, Data2VecVariant, Data2VecVariant.BASE)

        self.processor = None
        self.model = None

    def load(self, model_dir: str) -> None:
        """Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model files will be stored.

        """
        if self.model is not None:
            return

        checkpoint_filename = self.variant.checkpoint_filename
        checkpoint = os.path.join(model_dir, checkpoint_filename)

        if not os.path.exists(checkpoint):
            if (
                download_file(
                    f"https://dl.fbaipublicfiles.com/fairseq/data2vec/{checkpoint_filename}",
                    model_dir,
                )
                != checkpoint
            ):
                raise RuntimeError("Failed to download model checkpoint.")

        third_party_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "third_party")
        )
        if third_party_dir not in sys.path:
            sys.path.append(third_party_dir)

        from lfeats.third_party.fairseq.checkpoint_utils import (
            load_model_ensemble_and_task,
        )

        models, cfg, _ = load_model_ensemble_and_task([checkpoint])
        self.model = models[0]
        self.model.eval()
        self.model.to(self.device)

        self.normalize = cfg.task.normalize  # type: ignore

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

        if self.normalize:
            audio = audio.normalize()

        with torch.inference_mode():
            features = self.model(
                source=audio.tensor.to(self.device),
                mask=False,
                features_only=True,
            )["features"]

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
        variant_map = {
            Data2VecVariant.BASE: 12,
            Data2VecVariant.LARGE: 24,
        }
        return variant_map.get(self.variant, 0)

    @property
    def model_id(self) -> str:
        """Get the model identifier.

        Returns
        -------
        out : str
            The model identifier.

        """
        return f"data2vec-{self.variant.value}"
