# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""I/O utilities."""

from collections.abc import Generator
from contextlib import contextmanager

import torch
import torchaudio


@contextmanager
def temporary_hub_dir(path: str) -> Generator[None, None, None]:
    """Context manager to temporarily set the PyTorch Hub directory.

    Parameters
    ----------
    path : str
        The directory to set as the PyTorch Hub directory.

    """
    org_dir = torch.hub.get_dir()
    torch.hub.set_dir(path)
    try:
        yield
    finally:
        torch.hub.set_dir(org_dir)


def download_hf_file(**kwargs) -> str:
    """Download a file from Hugging Face Hub silently.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to `hf_hub_download`.

    Returns
    -------
    out : str
        The path to the downloaded file.

    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file.

    Parameters
    ----------
    path : str
        The path to the audio file.

    Returns
    -------
    out : tuple[torch.Tensor, int]
        A tuple containing the audio tensor and the sample rate.

    """
    return torchaudio.load(path, channels_first=True)
