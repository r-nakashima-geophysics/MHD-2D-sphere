"""A Python module to provide the utilities for debugging."""

import inspect
import sys
from types import FrameType

from package_common.default_logger import DefaultLogger
from package_common.utils_name import get_current_function_name


def under_construction_log() -> None:
    """Log the under construction message.

    Warnings
    --------
    Under construction
        If the feature is under construction.

    Examples
    --------
    >>> from package_common.utils_debug import under_construction_log
    >>> under_construction_log()
    """

    frame: FrameType | None = inspect.currentframe()

    function_name: str = get_current_function_name(frame)
    logger: DefaultLogger = DefaultLogger(function_name, level='INFO')

    logger.info('Under construction')
    sys.exit(0)
