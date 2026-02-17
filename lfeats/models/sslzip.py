# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the SSLZip model."""

from enum import Enum
from typing import cast

import numpy as np
import torch

from ..interfaces.types import Audio, Features
from ..utils.io import download_hf_file
from ..utils.validation import validate_enum
from .base import BaseModel
from .hubert import HubertModel, HubertVariant


class SslZipVariant(str, Enum):
    """Enumeration of supported SSLZip model variants."""

    TIGHT = "tight"
    LOOSE = "loose"

    @property
    def repo_and_filename(self) -> tuple[str, str]:
        """Return the repository and filename corresponding to the variant.

        Returns
        -------
        out : tuple[str, str]
            The repository and filename corresponding to the variant.

        """
        if self.value == "tight":
            return "takenori-y/SSLZip-16", "sslzip_16.onnx"
        elif self.value == "loose":
            return "takenori-y/SSLZip-256", "sslzip_256.onnx"
        raise ValueError(f"Unsupported SSLZip variant: {self.value}")


class SslZipModel(BaseModel):
    """A class for the SSLZip model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the SSLZip model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, SslZipVariant, SslZipVariant.LOOSE)

        self.upstream = HubertModel(variant=HubertVariant.BASE.value, device=device)
        self.model = None

    def load(self, model_dir: str) -> None:
        """Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model checkpoint will be stored.

        """
        if self.model is not None:
            return

        self.upstream.load(model_dir)

        repo_id, filename = self.variant.repo_and_filename
        model_path = download_hf_file(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=model_dir,
        )

        import onnxruntime as ort

        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.model = ort.InferenceSession(model_path, providers=providers)

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

        node_name = self.model.get_inputs()[0].name
        with torch.inference_mode():
            h = self.upstream.extract_features(audio, layers=[-1])
            z = self.model.run(None, {node_name: h.array})[0]
            z = cast(np.ndarray, z)

        return Features(data=z, source=self.model_id, layers=layers)

    @property
    def num_layers(self) -> int:
        """Get the number of available layers in the model.

        Returns
        -------
        out : int
            The number of layers.

        """
        return 1

    @property
    def model_id(self) -> str:
        """Get the model identifier.

        Returns
        -------
        out : str
            The model identifier.

        """
        return f"sslzip-{self.variant.value}"
