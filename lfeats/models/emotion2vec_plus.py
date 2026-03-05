# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for the emotion2vec+ model."""

import os
from enum import Enum

import torch
from huggingface_hub import hf_hub_download

from ..interfaces.types import Audio, Features
from ..utils.io import silence_hf_hub
from ..utils.paths import sanitize, setup_third_party_path
from ..utils.validation import validate_enum
from .base import FrameLevelFeatureModel


class Emotion2VecPlusVariant(str, Enum):
    """Enumeration of supported emotion2vec+ model variants."""

    SEED = "seed"
    BASE = "base"
    LARGE = "large"

    @property
    def repository(self) -> str:
        """Return the repository corresponding to the variant.

        Returns
        -------
        out : str
            The repository corresponding to the variant.

        """
        return f"emotion2vec/emotion2vec_plus_{self.value}"


class Emotion2VecPlusModel(FrameLevelFeatureModel):
    """A class for the emotion2vec+ model."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the emotion2vec+ model.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        super().__init__(variant, device)

        self.variant = validate_enum(
            variant, Emotion2VecPlusVariant, Emotion2VecPlusVariant.BASE
        )
        self._model_id = f"emotion2vec+-{self.variant.value}"

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
        if self.model is not None:
            return

        with silence_hf_hub(quiet):
            repo_id = self.variant.repository
            checkpoint = hf_hub_download(
                repo_id=repo_id,
                filename="model.pt",
                repo_type="model",
                local_dir=os.path.join(model_dir, sanitize(self.model_id)),
            )
            config = hf_hub_download(
                repo_id=repo_id,
                filename="config.yaml",
                repo_type="model",
                local_dir=os.path.join(model_dir, sanitize(self.model_id)),
            )

        from hyperpyyaml import load_hyperpyyaml

        with open(config) as f:
            cfg_from_repo = load_hyperpyyaml(f)

        arg_overrides = {
            "task": {"_name": "audio_pretraining"},
            "model": {"_name": "data2vec_multi", **cfg_from_repo["model_conf"]},
        }

        setup_third_party_path()

        from lfeats.third_party.fairseq.checkpoint_utils import (
            load_model_ensemble_and_task,
        )

        def remap_checkpoint_keys(state_dict):
            prefix = cfg_from_repo["scope_map"][0]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    new_key = k[len(prefix) :]
                    new_state_dict[new_key] = v
                elif k.startswith("proj."):
                    pass
                else:
                    new_state_dict[k] = v
            return new_state_dict

        models, cfg, _ = load_model_ensemble_and_task(
            [checkpoint],
            strict=True,
            arg_overrides=arg_overrides,
            remap=remap_checkpoint_keys,
            remove_pretraining_modules=True,
        )
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
                mode="AUDIO",
                mask=False,
                features_only=True,
            )["layer_results"]

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
        return 8
