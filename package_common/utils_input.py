"""A Python module to assist the input of parameters."""

import sys

from package_common.common_types import Callable, TypeVarIntFloat
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


def input_value(default: TypeVarIntFloat,
                cast: Callable[[str], TypeVarIntFloat]) \
        -> TypeVarIntFloat:
    """Input a value from the command line or use a default value.

    When there is a command line argument, it overrides the default
    value.

    Parameters
    ----------
    default : TypeVarIntFloat
        The default value.
    cast : Callable[[str], TypeVarIntFloat]
        A function to cast the command line argument.

    Returns
    -------
    TypeVarIntFloat
        The command line argument or the default value.

    Warnings
    --------
    Invalid argument
        If the command line argument is invalid.
    Too many input arguments
        If the command line arguments are too many.

    Examples
    --------
    Run a script without a command line argument:
        >>> from package_common.utils_input import input_value
        >>> input_value(1, int)
        1
        >>> input_value(1.0, float)
        1.0
    Run a script with a command line argument (say 2):
        >>> from package_common.utils_input import input_value
        >>> input_value(1, int)
        2
    """

    logger: DefaultLogger = create_function_name_logger()

    if len(sys.argv) == 2:
        try:
            return cast(sys.argv[1])
        except (ValueError, TypeError):
            logger.error('Invalid argument')
            sys.exit(1)

    elif len(sys.argv) > 2:
        logger.error('Too many input arguments')
        sys.exit(1)

    return default


def input_value_within(min_value: TypeVarIntFloat,
                       max_value: TypeVarIntFloat,
                       cast: Callable[[str], TypeVarIntFloat]) \
        -> TypeVarIntFloat:
    """Input a value within a specified range from the command line.

    Parameters
    ----------
    min_value : TypeVarIntFloat
        The minimum value of the specified range.
    max_value : TypeVarIntFloat
        The maximum value of the specified range.
    cast : Callable[[str], TypeVarIntFloat]
        A function to cast the command line argument.

    Returns
    -------
    chosen_value : TypeVarIntFloat
        A chosen value within the specified range.

    Warnings
    --------
    Quit
        If the character 'q' is input.
    Out of range
        If the input value is not within the specified range.
    Invalid input
        If the input string is not an integer or float.

    Examples
    --------
    >>> from package_common.utils_input import input_value_within
    >>> input_value_within(0, 10, int)
    Enter a value in [0, 10] or q to quit: 1
    1
    """

    logger: DefaultLogger = create_function_name_logger()

    input_str: str
    chosen_value: TypeVarIntFloat
    while True:
        input_str = input(
            f'Enter a value in [{min_value}, {max_value}] '
            + 'or q to quit: ').strip()

        if input_str.lower() == 'q':
            logger.info('Quit')
            sys.exit(0)

        try:
            chosen_value = cast(input_str)
            if min_value <= chosen_value <= max_value:
                return chosen_value
            logger.error('Out of range')
        except ValueError:
            logger.error('Invalid input')
