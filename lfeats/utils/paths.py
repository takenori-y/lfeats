# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Path utilities."""

import os
import sys


def setup_third_party_path(subdir: str | None = None) -> None:
    """Set up the path to the third-party directory.

    Parameters
    ----------
    subdir : str | None, optional
        An optional subdirectory within the third-party directory to add to the path.

    """
    path = os.path.join(os.path.dirname(__file__), "..", "third_party")
    if subdir is not None:
        path = os.path.join(path, subdir)
    third_party_dir = os.path.abspath(path)
    if third_party_dir not in sys.path:
        sys.path.insert(0, third_party_dir)
