# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""I/O utilities."""


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
