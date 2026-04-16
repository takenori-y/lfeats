# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the Spin model."""

import sys
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from types import ModuleType
from typing import cast

import torch
from huggingface_hub import hf_hub_download

from ..interfaces.types import Audio, Features
from ..utils.io import silence_hf_hub
from ..utils.validation import validate_enum
from .base import FrameLevelFeatureModel


@contextmanager
def lightning_mock_context() -> Generator[None, None, None]:
    """Context manager to mock PyTorch Lightning and its dependencies."""
    dummy_targets = [
        "pytorch_lightning",
        "pytorch_lightning.loggers.wandb",
    ]
    added_modules = []

    class UniversalMock(type):
        def __getattr__(cls, name):
            return cls

        def __call__(cls, *args, **kwargs):
            return cls

        def __iter__(cls):
            return iter([])

        @property
        def __path__(cls):
            return []

    class DummyClass(metaclass=UniversalMock):
        pass

    for target in dummy_targets:
        if target in sys.modules:
            continue
        try:
            import importlib.util

            spec = importlib.util.find_spec(target)
            if spec is None:
                raise ImportError
        except (ImportError, AttributeError, TypeError):
            sys.modules[target] = cast(ModuleType, DummyClass)
            added_modules.append(target)

    try:
        yield
    finally:
        for target in added_modules:
            if target in sys.modules:
                del sys.modules[target]


class SpinVariant(str, Enum):
    """Enumeration of supported Spin model variants.

    The number in the variant name indicates the codebook size used in the model.

    """

    HUBERT_128 = "hubert-128"
    HUBERT_256 = "hubert-256"
    HUBERT_512 = "hubert-512"
    HUBERT_1024 = "hubert-1024"
    HUBERT_2048 = "hubert-2048"
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
        return f"spin_{self.value.replace('-', '_')}.ckpt"


class SpinModel(FrameLevelFeatureModel):
    """A class for the Spin model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the Spin model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(variant, SpinVariant, SpinVariant.HUBERT_256)
        self._model_id = f"spin-{self.variant.value}"

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

        with silence_hf_hub(quiet):
            model_path = hf_hub_download(
                repo_id="vectominist/spin_ckpt",
                filename=self.variant.checkpoint_filename,
                repo_type="dataset",
                cache_dir=model_dir,
            )

        with lightning_mock_context():
            checkpoint = torch.load(
                model_path, map_location=torch.device("cpu"), weights_only=False
            )

        from lfeats.third_party.s3prl.util.download import set_dir, set_progress
        from lfeats.third_party.spin.model import SpinModel as _SpinModel
        from lfeats.third_party.spin.util import len_to_padding

        set_dir(model_dir)
        set_progress(not quiet)
        self.model = _SpinModel(checkpoint["hyper_parameters"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.len_to_padding = len_to_padding

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
            wavs_len = torch.LongTensor([wavs.shape[1]] * wavs.shape[0]).to(self.device)
            padding_mask = self.len_to_padding(cast(torch.LongTensor, wavs_len)).to(
                self.device
            )
            outputs = self.model((wavs, wavs_len, padding_mask), feat_only=True)
            vectors = torch.concat([outputs["feat_list"][i] for i in layers], dim=-1)

        return Features(data=vectors, source=self.model_id, layers=layers)
