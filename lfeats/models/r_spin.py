# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the R-Spin model."""

import os
from enum import Enum

import torch

from ..interfaces.types import Audio, Features
from ..utils.io import download_file
from ..utils.validation import validate_enum
from .base import FrameLevelFeatureModel


class RSpinVariant(str, Enum):
    """Enumeration of supported R-Spin model variants.

    The number in the variant name indicates the codebook size used in the model.

    """

    WAVLM_32 = "wavlm-32"
    WAVLM_64 = "wavlm-64"
    WAVLM_128 = "wavlm-128"
    WAVLM_256 = "wavlm-256"
    WAVLM_512 = "wavlm-512"
    WAVLM_1024 = "wavlm-1024"
    WAVLM_2048 = "wavlm-2048"

    @property
    def checkpoint_filename(self) -> str:
        """Returns the corresponding checkpoint filename for the variant.

        Returns
        -------
        out : str
            The checkpoint filename.

        """
        base, codebook_size = self.value.split("-")
        return f"{base}_rspin_{codebook_size}-40k.pt"


class RSpinModel(FrameLevelFeatureModel):
    """A class for the R-Spin model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the R-Spin model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, RSpinVariant, RSpinVariant.WAVLM_256)
        self._model_id = f"r-spin-{self.variant.value}"

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

        checkpoint = os.path.join(model_dir, self.variant.checkpoint_filename)

        if not os.path.exists(checkpoint):
            if (
                download_file(
                    (
                        "https://data.csail.mit.edu/public-release-sls/rspin/"
                        + self.variant.checkpoint_filename
                    ),
                    download_dir=model_dir,
                    quiet=quiet,
                )
                != checkpoint
            ):
                raise RuntimeError("Failed to download the model checkpoint.")

        from lfeats.third_party.rspin import RSpinWavlm

        self.model = RSpinWavlm.load_from_checkpoint(checkpoint)
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
            wavs = audio.tensor.to(self.device)
            feat_list, _, _ = self.model(wavs)
            vectors = torch.concat([feat_list[i] for i in layers], dim=-1)

        return Features(data=vectors, source=self.model_id, layers=layers)
