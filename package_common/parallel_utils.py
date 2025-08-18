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
        -> tuple[tuple[SharedMemory, ...],
                 list[tuple[str,
                      tuple[int, ...],
                      np.dtype]]]:
    """Create some shared memory arrays.

    Parameters
    ----------
    *arrays
        The tuple of arrays to be shared.
    name_prefix : str, optional, default 'array'
        The name prefix for the shared memory arrays.

    Returns
    -------
    tuple_shm : tuple[SharedMemory, ...]
        The tuple of shared memories.
    shared_info : list[tuple[str, tuple[int, ...], np.dtype]]
        The list of the information of the shared memory arrays.

    Warnings
    --------
    Invalid argument
        If the arguments are invalid.
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    shms: list[SharedMemory] = []
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

        shms.append(shm)
        shared_info.append(
            (shm.name, shared_array.shape, shared_array.dtype))

    tuple_shm: tuple[SharedMemory, ...] = tuple(shms)

    return tuple_shm, shared_info


def attach_shared_arrays(shared_info: list[tuple[str,
                                                 tuple[int, ...],
                                                 np.dtype]]) \
        -> tuple[tuple[SharedMemory, ...],
                 tuple[ArrayAny, ...]]:
    """Attach the shared memories in a subprocess.

    Parameters
    ----------
    shared_info : list[tuple[str, tuple[int, ...], np.dtype]]
        The list of the information of the shared memory arrays.

    Returns
    -------
    tuple_shm : tuple[SharedMemory, ...]
        The tuple of shared memories.
    tuple_shared_arrays : tuple[ArrayAny, ...]
        The tuple of shared memory arrays.
    """

    shms: list[SharedMemory] = []
    shared_arrays: list[ArrayAny] = []

    for name, shape, dtype in shared_info:
        shm = shared_memory.SharedMemory(name=name)
        shared_array \
            = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

        shms.append(shm)
        shared_arrays.append(shared_array)

    tuple_shm: tuple[SharedMemory, ...] = tuple(shms)
    tuple_shared_arrays: tuple[ArrayAny, ...] = tuple(shared_arrays)

    return tuple_shm, tuple_shared_arrays


def detach_shared_arrays(shms: tuple[SharedMemory, ...],
                         unlink: bool = False) -> None:
    """Detach the shared memories.

    Parameters
    ----------
    shms : tuple[SharedMemory, ...]
        The tuple of the shared memories.
    unlink : bool
        The boolean value to switch whether to unlink the shared
        memories or not.
    """

    for shm in shms:
        shm.close()

        if unlink:
            shm.unlink()
