# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the validation utilities."""

import numpy as np
import pytest
import torch

from lfeats.utils.validation import validate_length


@pytest.mark.parametrize("module", [np, torch])
def test_validate_length_pads_short_input(module) -> None:
    """Test if a short input is padded with zeros at the end."""
    x = module.ones((2, 3))
    y = validate_length(x, 5)
    assert isinstance(y, type(x))
    assert tuple(y.shape) == (2, 5)
    assert np.array_equal(np.asarray(y[:, :3]), np.ones((2, 3)))
    assert np.array_equal(np.asarray(y[:, 3:]), np.zeros((2, 2)))


@pytest.mark.parametrize("module", [np, torch])
def test_validate_length_keeps_long_input(module) -> None:
    """Test if an input with enough length is returned as is."""
    x = module.ones((2, 5))
    y = validate_length(x, 5)
    assert y is x
