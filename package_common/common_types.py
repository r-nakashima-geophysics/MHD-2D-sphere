"""A Python module to define type aliases."""

from typing import Any, Callable, Final, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from matplotlib import axes, figure

ArrayFloat: type = npt.NDArray[np.float64]
Figure: type = figure.Figure
Axes: type = axes.Axes

__all__ = [
    "Any",
    "Callable",
    "Final",
    "Optional",
    "TypeVar",
    "ArrayFloat",
    "Figure",
    "Axes",
]
