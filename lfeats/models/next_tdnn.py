# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the NeXt-TDNN model."""

import os
from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import download_file
from ..utils.paths import setup_third_party_path
from ..utils.validation import validate_enum
from .base import UtteranceLevelFeatureModel


class NeXtTDNNVariant(str, Enum):
    """Enumeration of supported NeXt-TDNN model variants."""

    LIGHT = "light"
    BASE = "base"
    BASE_V2 = "base-v2"

    @property
    def checkpoint_filename(self) -> tuple[str, str]:
        """Returns the corresponding checkpoint directory and filename for the variant.

        Returns
        -------
        out : tuple[str, str]
            The checkpoint directory and filename corresponding to the variant.

        """
        if self.value == "light":
            directory = "NeXt_TDNN_light_C256_B3_K65"
            filename = "NeXt_TDNN_light_C256_B3_K65.pt"
        elif self.value == "base":
            directory = "NeXt_TDNN_C256_B3_K65_7"
            filename = "NeXt_TDNN_C256_B3_K65_7.pt"
        elif self.value == "base-v2":
            directory = "NeXt_TDNN_C256_B3_K65_7_cyclical_lr_step"
            filename = "model.pt"
        else:
            raise ValueError(f"Unsupported NeXt-TDNN variant: {self.value}")
        return directory, filename


class NeXtTDNNModel(UtteranceLevelFeatureModel):
    """A class for the NeXt-TDNN model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the NeXt-TDNN model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, NeXtTDNNVariant, NeXtTDNNVariant.BASE_V2)
        self._model_id = f"next-tdnn-{self.variant.value}"

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

        directory, filename = self.variant.checkpoint_filename
        checkpoint = os.path.join(model_dir, filename)

        if not os.path.exists(checkpoint):
            if (
                download_file(
                    (
                        "https://raw.githubusercontent.com/dmlguq456/NeXt_TDNN_ASV"
                        f"/refs/heads/main/experiments/{directory}/{filename}"
                    ),
                    download_dir=model_dir,
                    quiet=quiet,
                )
                != checkpoint
            ):
                raise RuntimeError("Failed to download the model checkpoint.")

        setup_third_party_path("next_tdnn_asv")

        from lfeats.third_party.next_tdnn_asv.main import NeXtTDNNModel

        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))[
            "state_dict"
        ]
        new_state_dict = {
            (f"speaker_net.{k}" if not k.startswith("speaker_net.") else k): v
            for k, v in state_dict.items()
        }
        self.model = NeXtTDNNModel(directory)
        self.model.load_state_dict(new_state_dict)
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
            vectors = self.model(audio.tensor.to(self.device))
            vectors = vectors.unsqueeze(1)

        return Features(data=vectors, source=self.model_id)
