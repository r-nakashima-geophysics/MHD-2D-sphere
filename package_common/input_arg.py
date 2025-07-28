"""A Python module to assist the input of parameters"""

import sys

from package_common.default_logger import DefaultLogger


def input_int(int_default: int) -> int:
    """Input an integer

    When there is a command line argument, the default value of the
    integer is overwritten with the argument.

    Parameters
    ----------
    int_default : int
        The default value of the integer

    Returns
    ----------
    int
        The overwritten integer or the default value

    Warnings
    ----------
    Invalid argument
        If the command line argument is invalid.
    Too many input arguments
        If the command line arguments are too many.

    Examples
    ----------
    Run a script without a command line argument:
        >>> input_int(1)
        1
    Run a script with a command line argument (say 2):
        >>> input_int(1) 
        2
    """

    function_name: str = sys._getframe().f_code.co_name

    if len(sys.argv) == 2:
        try:
            return int(sys.argv[1])
        except ValueError:
            DefaultLogger(function_name).error('Invalid argument')
            sys.exit(1)

    elif len(sys.argv) > 2:
        DefaultLogger(function_name).error('Too many input arguments')
        sys.exit(1)

    return int_default


def input_float(float_default: float) -> float:
    """Input a floating point number

    When there is a command line argument, the default value of the
    floating point number is overwritten with the argument.

    Parameters
    ----------
    float_default : float
        The default value of the floating point number

    Returns
    ----------
    float
        The overwritten floating point number or the default value

    Warnings
    ----------
    Invalid argument
        If the command line argument is invalid.
    Too many input arguments
        If the command line arguments are too many.

    Examples
    ----------
    Run a script without a command line argument:
        >>> input_float(1.0)
        1.0
    Run a script with a command line argument (say 2.0):
        >>> input_float(1.0)
        2.0
    """

    function_name: str = sys._getframe().f_code.co_name

    if len(sys.argv) == 2:
        try:
            return float(sys.argv[1])
        except ValueError:
            DefaultLogger(function_name).error('Invalid argument')
            sys.exit(1)

    elif len(sys.argv) > 2:
        DefaultLogger(function_name).error('Too many input arguments')
        sys.exit(1)

    return float_default


def input_int_within(int_min: int,
                     int_max: int) -> int:
    """Input an integer within a specified range

    Parameters
    ----------
    int_min : int
        The minimum value of the specified range
    int_max : int
        The maximum value of the specified range

    Returns
    ----------
    int_chosen : int
        A chosen specified integer

    Warnings
    ----------
    Quit
        If the character 'q' is input.
    Out of range
        If the input integer is not within the specified range.
    Invalid input
        If the input character is not an integer.

    Examples
    ----------
    >>> input_int_within(0,10)
    (quit: q):  1
    1
    """

    int_chosen: int
    function_name: str = sys._getframe().f_code.co_name

    input_str: str
    while True:
        input_str = input('(quit: q):  ').lower().strip()

        if input_str == 'q':
            DefaultLogger(function_name).info('Quit')
            sys.exit(0)

        try:
            int_chosen = int(input_str)
            if int_min <= int_chosen <= int_max:
                return int_chosen
            else:
                DefaultLogger(function_name).error('Out of range')
        except ValueError:
            DefaultLogger(function_name).error('Invalid input')
