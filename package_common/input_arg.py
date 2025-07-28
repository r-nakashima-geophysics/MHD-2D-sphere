"""Assists the input of parameters"""

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def input_int(m_default: int) -> int:
    """Inputs a zonal wavenumber

    When there is a command line argument, the default value of the
    zonal wavenumber (order) is overwritten with its argument.

    Parameters
    ----------
    m_default : int
        A default value of the zonal wavenumber (order)

    Returns
    ----------
    m_order : int
        An overwritten zonal wavenumber (order)

    Raises
    ----------
    Invalid argument
        If the command line argument is invalid.
    Too many input arguments
        If the command line arguments are too many.

    Examples
    ----------
    When there is not a command line argument:
        python3
        >>> from package.input_arg import input_int
        >>> input_int(1)
        1
    When there is a command line argument:
        python3 - 2
        >>> from package.input_arg import input_int
        >>> input_int(1)
        2

    """

    m_order: int

    if len(sys.argv) == 2:
        arg1: str = sys.argv[1]

        if not (arg1.isdigit() and float(arg1).is_integer()):
            logger.error('Invalid argument')
            sys.exit()
        else:
            m_order = int(arg1)
        #

    elif len(sys.argv) > 2:
        logger.error('Too many input arguments')
        sys.exit()
    else:
        m_order = m_default
    #

    return m_order
#


def input_alpha(alpha_default: float) -> float:
    """Inputs the Lehnert number

    When there is a command line argument, the default value of the
    Lehnert number is overwritten with its argument.

    Parameters
    ----------
    alpha_default : float
        A default value of the Lehnert number

    Returns
    ----------
    alpha : float
        An overwritten Lehnert number

    Raises
    ----------
    Invalid argument
        If the command line argument is invalid.
    Too many input arguments
        If the command line arguments are too many.

    Examples
    ----------
    When there is not a command line argument:
        python3
        >>> from package.input_arg import input_alpha
        >>> input_alpha(1)
        1
    When there is a command line argument:
        python3 - 2
        >>> from package.input_arg import input_alpha
        >>> input_alpha(1)
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
        #
        return check
    #

    alpha: float

    if len(sys.argv) == 2:
        arg1: str = sys.argv[1]

        if not is_num(arg1):
            logger.error('Invalid argument')
            sys.exit()
        else:
            alpha = float(arg1)
        #
    #

    elif len(sys.argv) > 2:
        logger.error('Too many input arguments')
        sys.exit()
    else:
        alpha = alpha_default
    #

    return alpha
#


def input_int_within(min_int: int,
                     max_int: int) -> int:
    """Inputs an integer within an appropriate range

    Parameters
    ----------
    min_int : int
        The minimum value of an appropriate range of integers
    max_int : int
        The maximum value of an appropriate range of integers

    Returns
    ----------
    chosen_int : int
        A chosen appropriate integer

    Raises
    ----------
    Quit
        If you input 'q' to quit inputting numbers.
    Invalid integer
        If the inputted integer is not within the appropriate range.
    Invalid input
        If the inputted character is not an integer.

    Examples
    ----------
    >>> from package.input_arg import input_int_within
    >>> input_int_within(0,10)
    (quit: q):  1
    1
    >>> input_int_within(0,10)
    (quit: q):  q
    INFO:package.input_arg:Quit

    """

    chosen_int: int

    while True:
        input_str: str = input('(quit: q):  ')

        check_int: bool = False
        if input_str.isdigit() and float(input_str).is_integer():

            chosen_int = int(input_str)

            if min_int <= chosen_int <= max_int:
                break
            #

            check_int = True
        #

        if input_str == 'q':
            logger.info('Quit')
            sys.exit()
        #

        if check_int:
            logger.error('Invalid integer')
        else:
            logger.error('Invalid input')
        #
    #

    return chosen_int
#
