"""A Python module to provide the utilities for parallel processing."""

import inspect
import os
import sys
from multiprocessing import shared_memory

import numpy as np

from package_common.common_types import ArrayAny, Optional, SharedMemory
from package_common.default_logger import DefaultLogger


def set_num_threads(num_threads: int) -> None:
    """Set the number of threads for each process.

    Parameters
    ----------
    num_threads : int
        The number of threads for each process.

    Warnings
    --------
    Invalid argument
        If the arguments are invalid.
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    if num_threads <= 0:
        logger.error('Invalid argument')
        sys.exit(1)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['ACCELERATE_NTHREADS'] = str(num_threads)
    os.environ['BLIS_NUM_THREADS'] = str(num_threads)


def create_shared_arrays(*arrays,
                         name_prefix: Optional[str] = 'array') \
        -> list[tuple[str,
                      tuple[int, ...],
                      np.dtype]]:
    """Create some shared memory arrays.

    Parameters
    ----------
    *arrays
        The tuple of arrays to be shared.

    name_prefix : str, optional, default 'array'
        The name prefix for the shared memory.

    Returns
    -------
    shared_info : list[tuple[str, tuple[int, ...], np.dtype]]
        The list of the information of the shared memory.

    Warnings
    --------
    Invalid argument
        If the arguments are invalid.
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    shared_info: list[tuple[str, tuple[int, ...], np.dtype]] = []

    shm: SharedMemory
    shared_array: ArrayAny
    for i, array in enumerate(arrays):

        if not isinstance(array, np.ndarray):
            logger.error('Invalid argument')
            sys.exit(1)

        shm = shared_memory.SharedMemory(
            name=f'{name_prefix}_{i}', create=True, size=array.nbytes)
        shared_array = np.ndarray(shape=array.shape, dtype=array.dtype,
                                  buffer=shm.buf)
        shared_array[:] = array[:]

        shared_info.append(
            (shm.name, shared_array.shape, shared_array.dtype))

    return shared_info


def attach_shared_arrays(
        shared_info: list[tuple[str,
                                tuple[int, ...],
                                np.dtype]]) -> tuple[ArrayAny, ...]:
    """Attach the shared memory arrays.

    Parameters
    ----------
    shared_info : list[tuple[str, tuple[int, ...], np.dtype]]
        The list of the information of the shared memory.

    Returns
    -------
    tuple[ArrayAny, ...]
        The tuple of attached shared memory arrays.
    """

    shared_arrays: list[ArrayAny] = []

    for name, shape, dtype in shared_info:
        shm = shared_memory.SharedMemory(name=name)
        shared_array \
            = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

        shared_arrays.append(shared_array)

    return tuple(shared_arrays)


def detach_shared_arrays(shared_info: list[tuple[str,
                                                 tuple[int, ...],
                                                 np.dtype]],
                         unlink: bool = False) -> None:
    """Detach the shared memory arrays.

    Parameters
    ----------
    shared_info : list[tuple[str, tuple[int, ...], np.dtype]]
        The list of the information of the shared memory.
    unlink : bool
        The boolean value to switch whether to unlink the shared memory
        or not.
    """

    for name, _, _ in shared_info:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()

        if unlink:
            shm.unlink()
