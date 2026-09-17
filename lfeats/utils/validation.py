# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for common validation logic."""

from enum import Enum
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

EnumT = TypeVar("EnumT", bound=Enum)
ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


def validate_enum(value: Any, enum_class: type[EnumT], default: EnumT) -> EnumT:
    """Validate that the given value is a valid member of the specified enum class.

    Parameters
    ----------
    value : Any
        The value to validate.

    enum_class : type[Enum]
        The enum class to validate against.

    default : Enum
        The default value to return if the input value is None.

    Returns
    -------
    out : Enum
        The validated enum member corresponding to the input value.

    Raises
    ------
    ValueError
        If the input value is not a valid member of the enum class.

    """
    if value is None:
        return default
    try:
        return enum_class(value)
    except ValueError as e:
        supported = [v.value for v in enum_class]
        raise ValueError(
            f"Unsupported enum value '{value}'. Supported values are: {supported}"
        ) from e


def validate_length(x: ArrayT, min_length: int) -> ArrayT:
    """Validate that the input array has at least the specified minimum length.

    Parameters
    ----------
    x : np.ndarray | torch.Tensor
        The input array to validate.

    min_length : int
        The minimum expected length of the input array.

    Returns
    -------
    out : np.ndarray | torch.Tensor
        The input array, padded with zeros at the end of the last axis if its length
        is less than the minimum length.

    """
    actual_length = x.shape[-1]
    if min_length <= actual_length:
        return x
    if isinstance(x, np.ndarray):
        return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, min_length - actual_length)])
    return F.pad(x, (0, min_length - actual_length))
