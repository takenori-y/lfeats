# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""I/O utilities."""

from collections.abc import Generator
from contextlib import contextmanager

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import logging


@contextmanager
def silence_hf_hub() -> Generator[None, None, None]:
    """Context manager to silence Hugging Face Hub logging."""
    previous_verbosity = logging.get_verbosity()
    logging.set_verbosity_error()
    try:
        yield
    finally:
        logging.set_verbosity(previous_verbosity)


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
    with silence_hf_hub():
        return hf_hub_download(**kwargs)
