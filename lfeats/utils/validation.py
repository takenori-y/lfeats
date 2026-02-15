# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for common validation logic."""

from enum import Enum
from typing import Any, TypeVar

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
