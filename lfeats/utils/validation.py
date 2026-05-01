# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for common validation logic."""

from enum import Enum
from typing import Any, TypeVar

import torch
import torch.nn.functional as F

T = TypeVar("T", bound=Enum)


def validate_enum(value: Any, enum_class: type[T], default: T) -> T:
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


def validate_length(x: torch.Tensor, min_length: int) -> torch.Tensor:
    """Validate that the input tensor has at least the specified minimum length.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to validate.

    min_length : int
        The minimum expected length of the input tensor.

    Returns
    -------
    out : torch.Tensor
        The input tensor, padded with zeros if its length is less than the minimum
        length.

    """
    actual_length = x.shape[-1]
    if actual_length < min_length:
        x = F.pad(x, (0, min_length - actual_length))
    return x
