"""A Python module to define type aliases."""

from multiprocessing import shared_memory
from types import FrameType
from typing import Callable, Final, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from matplotlib import artist, axes, figure, legend

type SharedMemory = shared_memory.SharedMemory

type Figure = figure.Figure
type Axes = axes.Axes
type Legend = legend.Legend
type Artist = artist.Artist

type ArrayInt = npt.NDArray[np.int_]
type ArrayFloat = npt.NDArray[np.float64]
type ArrayComplex = npt.NDArray[np.complex128]
type ArrayBool = npt.NDArray[np.bool_]
type ArrayStr = npt.NDArray[np.str_]
type ArrayAny = npt.NDArray[np.object_]

type ArrayAxes = npt.NDArray[np.object_]

type ComplexFunc = Callable[[complex], complex]

TypeVarIntFloat = TypeVar('TypeVarIntFloat', int, float)
TypeVarFloatComplex = TypeVar('TypeVarFloatComplex', float, complex)
