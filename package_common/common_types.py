"""A Python module to define type aliases."""

from typing import Any, Callable, Final, Optional, TypeVar

import numpy as np
import numpy.typing as npt

ArrayFloat: type = npt.NDArray[np.float64]

__all__ = [
    "Any",
    "Callable",
    "Final",
    "Optional",
    "TypeVar",
    "ArrayFloat",
]
