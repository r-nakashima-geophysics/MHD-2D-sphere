"""A Python module to define type aliases."""

from typing import Final

import numpy as np
import numpy.typing as npt

ArrayFloat = npt.NDArray[np.float64]

__all__ = [
    "Final",
    "ArrayFloat",
]
