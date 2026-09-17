# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the DAC-VAE model."""

from enum import Enum

import torch
from huggingface_hub import hf_hub_download

from ..interfaces.types import Audio, Features
from ..utils.io import silence_hf_hub
from ..utils.validation import validate_enum, validate_length
from .base import TokenLevelFeatureModel


class DACVAEVariant(str, Enum):
    """Enumeration of supported DAC-VAE model variants."""

    BASE = "base"

    @property
    def model_name(self) -> str:
        """Return the model name corresponding to the variant.

        Returns
        -------
        out : str
            The model name corresponding to the variant.

        """
        return "facebook/dacvae-watermarked"


class DACVAEModel(TokenLevelFeatureModel):
    """A class for the DAC-VAE model.

    The decoder of the original model is not constructed because only the encoder is
    required to extract features. This also means that the watermarking module, which
    is a part of the decoder, is never used. In addition, the loudness normalization
    performed in the original inference script is not applied.

    """

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the DAC-VAE model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, DACVAEVariant, DACVAEVariant.BASE)
        self._model_id = f"dacvae-{self.variant.value}"

    def load(self, model_dir: str, quiet: bool = False) -> None:
        """Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model checkpoint will be stored.

        quiet : bool, optional
            Whether to suppress output during the loading process.

        Raises
        ------
        RuntimeError
            If the model checkpoint does not contain the expected parameters.

        """
        if self.model is not None:
            return

        with silence_hf_hub(quiet):
            model_path = hf_hub_download(
                repo_id=self.variant.model_name,
                filename="weights.pth",
                repo_type="model",
                cache_dir=model_dir,
            )

        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )

        from lfeats.third_party.dacvae import DACVAE

        self.model = DACVAE(**checkpoint["metadata"]["kwargs"])

        # The checkpoint contains the parameters of the decoder, which are not used.
        missing_keys, _ = self.model.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        if missing_keys:
            raise RuntimeError(f"Missing parameters in the checkpoint: {missing_keys}")

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
            raise RuntimeError("Model not loaded. Call 'load' method first.")

        with torch.inference_mode():
            inputs = audio.tensor.to(self.device)
            # The model pads the input in the reflect mode, which requires the padding
            # size to be smaller than the input length.
            inputs = validate_length(inputs, self.frame_shift // 2 + 1)
            vectors = self.model.encode(inputs.unsqueeze(1))  # (B, D, N)
            vectors = vectors.transpose(1, 2)

        return Features(data=vectors, source=self.model_id)

    @property
    def frame_shift(self) -> int:
        """Get the frame shift of the model.

        Returns
        -------
        out : int
            The frame shift in samples.

        """
        return int(40.0 * self.sample_rate / 1000)

    @property
    def sample_rate(self) -> int:
        """Get the sample rate required by the model.

        Returns
        -------
        out : int
            The sample rate in Hz.

        """
        return 48000
