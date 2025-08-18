"""A Python module to provide the utilities for parallel processing."""

import inspect
import os
import sys

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

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["ACCELERATE_NTHREADS"] = str(num_threads)
    os.environ["BLIS_NUM_THREADS"] = str(num_threads)
