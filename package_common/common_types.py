"""A Python module to define type aliases."""

from multiprocessing import shared_memory
from types import FrameType
from typing import Any, Callable, Final, Optional, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from matplotlib import artist, axes, figure, legend

SharedMemory: TypeAlias = shared_memory.SharedMemory

Figure: TypeAlias = figure.Figure
Axes: TypeAlias = axes.Axes
Legend: TypeAlias = legend.Legend
Artist: TypeAlias = artist.Artist

ArrayInt: TypeAlias = npt.NDArray[np.int_]
ArrayFloat: TypeAlias = npt.NDArray[np.float64]
ArrayComplex: TypeAlias = npt.NDArray[np.complex128]
ArrayStr: TypeAlias = npt.NDArray[np.str_]
ArrayAny: TypeAlias = npt.NDArray[Any]

ArrayAxes: TypeAlias = npt.NDArray[np.object_]

ComplexFunc: TypeAlias = Callable[[complex], complex]
