# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""I/O utilities."""

from collections.abc import Generator
from contextlib import contextmanager

import torch
import torchaudio


@contextmanager
def set_torch_hub_dir(path: str) -> Generator[None, None, None]:
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


@contextmanager
def silence_transformers(enabled: bool = True) -> Generator[None, None, None]:
    """Context manager to silence the progress bar from Transformers.

    Parameters
    ----------
    enabled : bool, optional
        Whether to silence the progress bar. Default is True.

    """
    from transformers.utils.logging import disable_progress_bar, enable_progress_bar

    if enabled:
        disable_progress_bar()
    try:
        yield
    finally:
        if enabled:
            enable_progress_bar()


@contextmanager
def silence_hf_hub(enabled: bool = True) -> Generator[None, None, None]:
    """Context manager to silence Hugging Face Hub logging.

    Parameters
    ----------
    enabled : bool, optional
        Whether to silence Hugging Face Hub logging.

    """
    from huggingface_hub.utils.tqdm import disable_progress_bars, enable_progress_bars

    if enabled:
        disable_progress_bars()
    try:
        yield
    finally:
        if enabled:
            enable_progress_bars()


def download_file(url: str, download_dir: str, quiet: bool = False) -> str:
    """Download a file from a URL.

    Parameters
    ----------
    url : str
        The URL of the file to download.

    download_dir : str
        The directory to download the file to.

    quiet : bool, optional
        Whether to suppress the download progress output.

    Returns
    -------
    out : str
        The path to the downloaded file.

    Raises
    ------
    RuntimeError
        If the file could not be downloaded.

    """
    from parfive import Downloader

    dl = Downloader(progress=not quiet)

    dl.enqueue_file(url, path=download_dir)

    results = dl.download()
    if len(results) == 0:
        raise RuntimeError(f"Failed to download file: {results.errors}")
    return results[0]


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
