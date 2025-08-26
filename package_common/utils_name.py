"""A Python module to provide the utilities for naming."""

import inspect
import logging
import sys
from types import FrameType

from package_common.default_logger import DefaultLogger
from package_common.progress_bar import ProgressBar


def get_current_function_name(frame: FrameType | None = None) -> str:
    """Return the name of the current function.

    Parameters
    ----------
    frame : FrameType | None, optional, default None
        The frame to inspect.

    Returns
    -------
    function_name : str
        The name of the current function.

    Warnings
    --------
    Invalid type of the argument.
        If the argument is not a FrameType.

    Notes
    -----
    If a frame is not input in the argument of this function, the name
    of the function which calls this function will be returned. If a
    frame is input, the name of the function which calls the function
    obtaining the frame will be returned.

    Examples
    --------
    >>> from package_common.utils_name import get_current_function_name
    >>> def func():
    ...     return get_current_function_name()
    ...
    >>> func()
    'func'
    """

    if frame is None:
        frame = inspect.currentframe()

    this_frame: FrameType | None = inspect.currentframe()
    this_name: str
    if this_frame is not None:
        this_name = this_frame.f_code.co_name
    logger: DefaultLogger = DefaultLogger(this_name)
    if not isinstance(frame, FrameType):
        logger.error('Invalid type of the argument')
        sys.exit(1)

    function_name: str = 'Unknown'
    if (frame is not None) and (frame.f_back is not None):
        function_name = frame.f_back.f_code.co_name

    return function_name


def create_function_name_logger(level: int | str = logging.INFO) \
        -> DefaultLogger:
    """Create a logger with the name of the current function.

    Parameters
    ----------
    level : int, optional, default logging.INFO
        The logging level.

    Returns
    -------
    DefaultLogger
        The logger instance.

    Examples
    --------
    >>> from package_common.utils_name import
    create_function_name_logger
    >>> logger = create_function_name_logger()
    """

    frame: FrameType | None = inspect.currentframe()
    function_name: str = get_current_function_name(frame)

    return DefaultLogger(function_name, level=level)


def create_function_name_progress_bar(num_calc: int) -> ProgressBar:
    """Create a progress bar with the name of the current function.

    Parameters
    ----------
    num_calc : int
        The total iteration number.

    Returns
    -------
    ProgressBar
        The instance of the ProgressBar class.

    Examples
    --------
    >>> from package_common.progress_bar import
    create_function_name_progress_bar
    >>> progress_bar = create_function_name_progress_bar(100)
    """

    frame: FrameType | None = inspect.currentframe()
    function_name: str = get_current_function_name(frame)

    return ProgressBar(function_name, num_calc)
