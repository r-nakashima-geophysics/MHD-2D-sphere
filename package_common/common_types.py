"""A Python module to define type aliases.

Examples
--------
>>> from package_common.common_types import ArrayFloat
>>> import numpy as np
>>> array: ArrayFloat = np.array([1.0, 2.0, 3.0])
>>> print(array)
[1. 2. 3.]
"""

from multiprocessing import shared_memory
from typing import Callable, Final, TypeVar

import numpy as np
import numpy.typing as npt

type SharedMemory = shared_memory.SharedMemory

type ArrayInt = npt.NDArray[np.int_]
type ArrayFloat = npt.NDArray[np.float64]
type ArrayComplex = npt.NDArray[np.complex128]
type ArrayBool = npt.NDArray[np.bool_]
type ArrayStr = npt.NDArray[np.str_]
type ArrayAny = npt.NDArray[np.object_]

type ComplexFunc = Callable[[complex], complex]

TypeVarIntFloat = TypeVar('TypeVarIntFloat', int, float)
TypeVarFloatComplex = TypeVar('TypeVarFloatComplex', float, complex)
