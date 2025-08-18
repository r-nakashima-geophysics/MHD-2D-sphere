"""A Python module to define type aliases."""

from multiprocessing import shared_memory
from typing import Any, Callable, Final, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from matplotlib import artist, axes, figure, legend

SharedMemory: type = shared_memory.SharedMemory

Figure: type = figure.Figure
Axes: type = axes.Axes
Legend: type = legend.Legend
Artist: type = artist.Artist

ArrayInt: type = npt.NDArray[np.int_]
ArrayFloat: type = npt.NDArray[np.float64]
ArrayComplex: type = npt.NDArray[np.complex128]
ArrayStr: type = npt.NDArray[np.str_]
ArrayAny: type = npt.NDArray[Any]
ArrayAxes: type = npt.NDArray[Axes]

ComplexFunc: type = Callable[[complex], complex]

__all__ = [
    'SharedMemory',
    'Any',
    'Callable',
    'Final',
    'Optional',
    'TypeVar',
    'Figure',
    'Axes',
    'Legend',
    'Artist',
    'ArrayInt',
    'ArrayFloat',
    'ArrayComplex',
    'ArrayStr',
    'ArrayAny',
    'ArrayAxes',
    'ComplexFunc',
]
