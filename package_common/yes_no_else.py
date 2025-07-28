"""A decorator to decide whether to execute a function"""

import logging
import sys
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def yes_exe_no_exit(func) -> Callable[..., None]:
    """A decorator to execute a function when you input 'yes' and to
    exit the program when you input 'no'

    Parameters
    ----------
    func : Callable
        A function executed when you input 'yes'

    Returns
    ----------
    new_function : Callable
        A function executed when you input 'yes'

    Raises
    ----------
    Quit
        If you input 'n' or 'no'.
    Invalid input
        If you input characters other than 'y', 'yes', 'n', or 'no'.

    Examples
    ----------
    >>> from package.yes_no_else import yes_exe_no_exit
    >>> def test():
    ...     print('test')
    ...
    >>> @yes_exe_no_exit
    ... def wrapper():
    ...     test()
    ...
    >>> wrapper()
    (yes/no) yes
    test

    """

    def new_function(*args, **kwargs) -> None:

        yes_no: str

        while True:
            yes_no = input('(yes/no) ').lower()

            if yes_no in ('y', 'yes'):
                func(*args, **kwargs)
                break
            #

            if yes_no in ('n', 'no'):
                logger.info('Quit')
                sys.exit()
            #

            logger.error('Invalid input')
        #
    #

    return new_function
#


def exe_yes_continue(func) -> Callable[..., None]:
    """A decorator to execute a function, and then to re-execute a
    function only when you input 'yes'

    Parameters
    ----------
    func :
        A function executed when you input 'yes'

    Returns
    ----------
    new_function : Callable
        A function executed when you input 'yes'

    Raises
    ----------
    Quit
        If you input 'n' or 'no'.
    Invalid input
        If you input characters other than 'y', 'yes', 'n', or 'no'.

    Examples
    ----------
    >>> from package.yes_no_else import exe_yes_continue
    >>> def test():
    ...     print('test')
    ...
    >>> @exe_yes_continue
    ... def wrapper():
    ...     test()
    ...
    >>> wrapper()
    test
    Re-execute? (yes/no) yes
    test
    Re-execute? (yes/no) no
    INFO:package.yes_no_else:Quit

    """

    def new_function(*args, **kwargs) -> None:

        yes_no: str = 'y'
        may_be_invalid: bool = False

        while True:

            if yes_no in ('y', 'yes'):
                may_be_invalid = False
                func(*args, **kwargs)
            #

            if yes_no in ('n', 'no'):
                logger.info('Quit')
                sys.exit()
            #

            if may_be_invalid:
                logger.error('Invalid input')
            #

            may_be_invalid = True
            yes_no = input('Re-execute? (yes/no) ').lower()
        #
    #

    return new_function
#
