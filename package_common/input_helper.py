"""A Python module to assist the input of parameters."""

import inspect
import sys

from package_common.common_types import Callable, TypeVar
from package_common.default_logger import DefaultLogger

T = TypeVar("T", int, float)


def input_value(default: T,
                cast: Callable[[str], T]) -> T:
    """Input a value from the command line or use a default value.

    When there is a command line argument, it overrides the default
    value.

    Parameters
    ----------
    default : T
        The default value.
    cast : Callable[[str], T]
        A function to cast the command line argument.

    Returns
    ----------
    T
        The command line argument or the default value.

    Warnings
    ----------
    Invalid argument
        If the command line argument is invalid.
    Too many input arguments
        If the command line arguments are too many.

    Examples
    ----------
    Run a script without a command line argument:
        >>> input_value(1, int)
        1
        >>> input_value(1.0, float)
        1.0
    Run a script with a command line argument (say 2):
        >>> input_value(1, int)
        2
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(name=function_name)

    if len(sys.argv) == 2:
        try:
            return cast(sys.argv[1])
        except ValueError:
            logger.error('Invalid argument')
            sys.exit(1)

    elif len(sys.argv) > 2:
        logger.error('Too many input arguments')
        sys.exit(1)

    return default


def input_value_within(min_value: T,
                       max_value: T,
                       cast: Callable[[str], T]) -> T:
    """Input a value within a specified range from the command line.

    Parameters
    ----------
    min_value : T
        The minimum value of the specified range
    max_value : T
        The maximum value of the specified range
    cast : Callable[[str], T]
        A function to cast the command line argument.

    Returns
    ----------
    chosen_value : T
        A chosen value within the specified range.

    Warnings
    ----------
    Quit
        If the character 'q' is input.
    Out of range
        If the input value is not within the specified range.
    Invalid input
        If the input character is not an integer.

    Examples
    ----------
    >>> input_value_within(0, 10, int)
    (quit: q):  1
    1
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(name=function_name)

    input_str: str
    chosen_value: T
    while True:
        input_str = input('(quit: q):  ').strip().lower()

        if input_str == 'q':
            logger.info('Quit')
            sys.exit(0)

        try:
            chosen_value = cast(input_str)
            if min_value <= chosen_value <= max_value:
                return chosen_value
            logger.error('Out of range')
        except ValueError:
            logger.error('Invalid input')
