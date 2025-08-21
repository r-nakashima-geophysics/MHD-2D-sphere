"""A Python module to provide the utilities for debugging."""

import inspect
import sys

from package_common.common_types import FrameType, Optional
from package_common.default_logger import DefaultLogger
from package_common.name_utils import get_current_function_name


def under_construction_log() -> None:
    """Log the under construction message.

    Examples
    --------
    >>> under_construction_log()
    """

    frame: Optional[FrameType] = inspect.currentframe()

    function_name: str = get_current_function_name(frame)
    del frame
    logger: DefaultLogger = DefaultLogger(function_name, level='DEBUG')

    logger.debug('Under construction')
    sys.exit(0)
