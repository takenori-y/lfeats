# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""I/O utilities."""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import soundfile as sf
import torch
import torchaudio
from filelock import FileLock

HF_HTTP_LOGGER = "huggingface_hub.utils._http"


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
        Whether to silence the progress bar.

    """
    from transformers.utils.logging import disable_progress_bar, enable_progress_bar

    org_level = logging.getLogger(HF_HTTP_LOGGER).level
    if enabled:
        logging.getLogger(HF_HTTP_LOGGER).setLevel(logging.ERROR)
        disable_progress_bar()
    try:
        yield
    finally:
        if enabled:
            logging.getLogger(HF_HTTP_LOGGER).setLevel(org_level)
            enable_progress_bar()


@contextmanager
def silence_hf_hub(enabled: bool = True) -> Generator[None, None, None]:
    """Context manager to silence the progress bars from Hugging Face Hub.

    Parameters
    ----------
    enabled : bool, optional
        Whether to silence the progress bars.

    """
    from huggingface_hub.utils.tqdm import disable_progress_bars, enable_progress_bars

    org_level = logging.getLogger(HF_HTTP_LOGGER).level
    if enabled:
        logging.getLogger(HF_HTTP_LOGGER).setLevel(logging.ERROR)
        disable_progress_bars()
    try:
        yield
    finally:
        if enabled:
            logging.getLogger(HF_HTTP_LOGGER).setLevel(org_level)
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

    basename = os.path.basename(url)
    lock_path = os.path.join(download_dir, f"{basename}.lock")

    with FileLock(lock_path):
        dl = Downloader(progress=not quiet)

        dl.enqueue_file(url, path=download_dir)

        results = dl.download()
        if len(results) == 0:
            raise RuntimeError(f"Failed to download file: {results.errors}")
        return results[0]


def safe_torch_hub_load(
    repo_or_dir: str, model: str, download_dir: str, quiet: bool = False, **kwargs
) -> Any:
    """Safely load a model from PyTorch Hub.

    Parameters
    ----------
    repo_or_dir : str
        The repository or directory to load the model from.

    model : str
        The name of the model defined in the repository.

    download_dir : str
        The directory to use for caching the downloaded model files.

    quiet : bool, optional
        Whether to suppress output during the loading process.

    **kwargs : Any
        Additional keyword arguments to pass to `torch.hub.load`.

    Returns
    -------
    out : Any
        The loaded object.

    """
    lock_path = os.path.join(download_dir, f"{model}.lock")

    with FileLock(lock_path):
        with set_torch_hub_dir(download_dir):
            return torch.hub.load(
                repo_or_dir, model, verbose=not quiet, trust_repo="check", **kwargs
            )


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
    try:
        version("torchcodec")
        x, sr = torchaudio.load(path, channels_first=True)
    except PackageNotFoundError:
        x, sr = sf.read(path)
        x = torch.tensor(x.T, dtype=torch.float32)
    return x, sr
