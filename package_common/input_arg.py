"""A Python module to assist the input of parameters"""

import sys

from package_common.default_logger import DefaultLogger


def input_int(integer_default: int) -> int:
    """Input an integer

    When there is a command line argument, the default value of the
    integer is overwritten with the argument.

    Parameters
    ----------
    integer_default : int
        The default value of the integer

    Returns
    ----------
    integer : int
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
        $ python3
        >>> from package_common.input_arg import input_int
        >>> input_int(1)
        1

    Run a script with a command line argument:
        $ python3 - 2 
        >>> from package_common.input_arg import input_int 
        >>> input_int(1) 
        2
    """

    function_name: str = sys._getframe().f_code.co_name
    integer: int

    if len(sys.argv) == 2:
        arg1: str = sys.argv[1]

        if not (arg1.isdigit() and float(arg1).is_integer()):
            DefaultLogger(function_name).error('Invalid argument')
            sys.exit()
        else:
            integer = int(arg1)

    elif len(sys.argv) > 2:
        DefaultLogger(function_name).error('Too many input arguments')
        sys.exit()
    else:
        integer = integer_default

    return integer


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
    floating_point_num : float
        The overwritten floating point number or the default value

    Raises
    ----------
    Invalid argument
        If the command line argument is invalid.
    Too many input arguments
        If the command line arguments are too many.

    Examples
    ----------
    Run a script without a command line argument:
        $ python3 
        >>> from package_common.input_arg import input_float 
        >>> input_float(1.0)
        1.0

    Run a script with a command line argument:
        $ python3 - 2
        >>> from package_common.input_arg import input_float
        >>> input_float(1.0)
        2.0
    """

    def is_num(input_str) -> bool:
        check: bool
        try:
            float(input_str)
        except ValueError:
            check = False
        else:
            check = True

        return check

    function_name: str = sys._getframe().f_code.co_name
    floating_point_num: float

    if len(sys.argv) == 2:
        arg1: str = sys.argv[1]

        if not is_num(arg1):
            DefaultLogger(function_name).error('Invalid argument')
            sys.exit()
        else:
            floating_point_num = float(arg1)

    elif len(sys.argv) > 2:
        DefaultLogger(function_name).error('Too many input arguments')
        sys.exit()
    else:
        floating_point_num = float_default

    return floating_point_num


def input_int_within(integer_min: int,
                     integer_max: int) -> int:
    """Input an integer within an specified range

    Parameters
    ----------
    integer_min : int
        The minimum value of an specified range of integers
    integer_max : int
        The maximum value of an specified range of integers

    Returns
    ----------
    integer_chosen : int
        A chosen specified integer

    Warnings
    ----------
    Quit
        If the character 'q' is inputted.
    Invalid integer
        If the inputted integer is not within the specified range.
    Invalid input
        If the inputted character is not an integer.

    Examples
    ----------
    >>> from package_common.input_arg import input_int_within
    >>> input_int_within(0,10)
    (quit: q):  1
    1
    """

    integer_chosen: int
    function_name: str = sys._getframe().f_code.co_name

    input_str: str
    check_int: bool

    while True:
        input_str = input('(quit: q):  ')

        check_int = False
        if input_str.isdigit() and float(input_str).is_integer():

            integer_chosen = int(input_str)

            if integer_min <= integer_chosen <= integer_max:
                break

            check_int = True

        if input_str == 'q':
            DefaultLogger(function_name).info('Quit')
            sys.exit()

        if check_int:
            DefaultLogger(function_name).error('Invalid integer')
        else:
            DefaultLogger(function_name).error('Invalid input')

    return integer_chosen
