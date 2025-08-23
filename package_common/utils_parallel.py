"""A Python module to provide the utilities for parallel processing."""

import os
import sys
from multiprocessing import shared_memory

import numpy as np
import psutil

from package_common.common_types import ArrayAny, SharedMemory
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


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

    Examples
    --------
    Run a script with 4 threads:
        >>> from package_common.utils_parallel import set_num_threads
        >>> set_num_threads(4)
    """

    logger: DefaultLogger = create_function_name_logger()

    if num_threads <= 0:
        logger.error('Invalid argument')
        sys.exit(1)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['ACCELERATE_NTHREADS'] = str(num_threads)
    os.environ['BLIS_NUM_THREADS'] = str(num_threads)


def set_num_process() -> int:
    """Set the number of processes for multiprocessing.

    Returns
    -------
    int
        The number of processes for multiprocessing.

    Examples
    --------
    >>> from package_common.utils_parallel import set_num_process
    >>> set_num_process()
    4
    """

    num_process_physical: int | None = psutil.cpu_count(logical=False)
    num_process_logical: int | None = psutil.cpu_count(logical=True)

    if (num_process_physical is None) or (num_process_logical is None):
        return 1

    if num_process_physical == 1:
        return 1

    if num_process_physical == num_process_logical:
        return num_process_physical - 1

    return num_process_physical


def create_shared_arrays(*arrays) \
        -> tuple[tuple[SharedMemory, ...],
                 list[tuple[str,
                      tuple[int, ...],
                      np.dtype]]]:
    """Create some shared memory arrays.

    Parameters
    ----------
    *arrays
        The tuple of arrays to be shared.

    Returns
    -------
    tuple_shm : tuple[SharedMemory, ...]
        The tuple of shared memories.
    shared_info : list[tuple[str, tuple[int, ...], np.dtype]]
        The list of the information of the shared memory arrays.

    Warnings
    --------
    Invalid type of the argument
        If the argument is not a np.ndarray.

    Examples
    --------
    In the main process:
        >>> from package_common.utils_parallel import
        create_shared_arrays
        >>> shm, info = create_shared_arrays(np.array([1, 2, 3]))
    """

    logger: DefaultLogger = create_function_name_logger()

    shms: list[SharedMemory] = []
    shared_info: list[tuple[str, tuple[int, ...], np.dtype]] = []

    shm: SharedMemory
    shared_array: ArrayAny
    for array in arrays:

        if not isinstance(array, np.ndarray):
            logger.error('Invalid type of the argument')
            sys.exit(1)

        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
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

    Examples
    --------
    In a subprocess:
        >>> from package_common.utils_parallel import
        attach_shared_arrays
        >>> shm, array = attach_shared_arrays(info)
    """

    shms: list[SharedMemory] = []
    shared_arrays: list[ArrayAny] = []

    shm: SharedMemory
    shared_array: ArrayAny
    for name, shape, dtype in shared_info:
        shm = shared_memory.SharedMemory(name=name)
        shared_array \
            = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

        shms.append(shm)
        shared_arrays.append(shared_array)

    tuple_shm: tuple[SharedMemory, ...] = tuple(shms)
    tuple_shared_arrays: tuple[ArrayAny, ...] = tuple(shared_arrays)

    return tuple_shm, tuple_shared_arrays


def detach_shared_arrays(*shms,
                         unlink: bool = False) -> None:
    """Detach the shared memories.

    Parameters
    ----------
    *shms
        The tuple of the shared memories.
    unlink : bool, optional, default False
        The boolean value to switch whether to unlink the shared
        memories or not.

    Warnings
    --------
    Shared memory has already been unlinked
        If the shared memory has already been unlinked.

    Examples
    --------
    In a subprocess:
        >>> from package_common.utils_parallel import
        detach_shared_arrays
        >>> detach_shared_arrays(shm)
    In the main process:
        >>> from package_common.utils_parallel import
        detach_shared_arrays
        >>> detach_shared_arrays(shm, unlink=True)
    """

    logger: DefaultLogger = create_function_name_logger()

    for shm in shms:
        shm.close()

        if unlink:
            try:
                shm.unlink()
            except FileNotFoundError:
                logger.warning(
                    'Shared memory has already been unlinked')
