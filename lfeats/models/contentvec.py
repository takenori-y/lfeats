# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the ContentVec model."""

import os
import sys
from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import download_file
from ..utils.validation import validate_enum
from .base import BaseModel


class ContentVecVariant(str, Enum):
    """Enumeration of supported ContetVec model variants.

    The number in the variant name indicates the number of classes used in the model.

    """

    HUBERT_100 = "hubert-100"
    HUBERT_500 = "hubert-500"

    @property
    def token_and_checkpoint_filename(self) -> tuple[str, str]:
        """Returns the corresponding checkpoint filename for the variant.

        Returns
        -------
        out : tuple[str, str]
            The token and checkpoint filename corresponding to the variant.

        """
        _, num_classes = self.value.split("-")
        filename = f"checkpoint_best_legacy_{num_classes}.pt"
        if self.value == "hubert-100":
            token = "t76fff0dciyjqt1db03y48323qp99bg9"
        elif self.value == "hubert-500":
            token = "z1wgl1stco8ffooyatzdwsqn2psd9lrr"
        else:
            raise ValueError(f"Unsupported ContentVec variant: {self.value}")
        return token, filename


class ContentVecModel(BaseModel):
    """A class for the ContetVec model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the ContentVec model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(
            variant, ContentVecVariant, ContentVecVariant.HUBERT_100
        )

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

        token, checkpoint_filename = self.variant.token_and_checkpoint_filename
        checkpoint = os.path.join(model_dir, checkpoint_filename)

        if not os.path.exists(checkpoint):
            if (
                download_file(
                    f"https://ibm.ent.box.com/shared/static/{token}",
                    download_dir=model_dir,
                    quiet=quiet,
                )
                != checkpoint
            ):
                raise RuntimeError("Failed to download the model checkpoint.")

        third_party_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "third_party")
        )
        if third_party_dir not in sys.path:
            sys.path.append(third_party_dir)

        from lfeats.third_party.fairseq.checkpoint_utils import (
            load_model_ensemble_and_task,
        )

        models, _, _ = load_model_ensemble_and_task([checkpoint])
        self.model = models[0]
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
            features = self.model(
                source=audio.tensor.to(self.device),
                mask=False,
                features_only=True,
            )["features"]

            vectors = torch.concat([features[i] for i in layers], dim=-1)

        return Features(data=vectors, source=self.model_id, layers=layers)

    @property
    def model_id(self) -> str:
        """Get the model identifier.

        Returns
        -------
        out : str
            The model identifier.

        """
        return f"rspin-{self.variant.value}"
