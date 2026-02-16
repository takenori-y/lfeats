# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Pytest configuration."""

from collections.abc import Generator

import pytest
from transformers.utils.logging import disable_progress_bar, enable_progress_bar


@pytest.fixture(autouse=True)
def silence_transformers_fixture() -> Generator[None, None, None]:
    """Hide the progress bar from Transformers during tests."""
    disable_progress_bar()
    yield
    enable_progress_bar()
