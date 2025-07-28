"""A Python module to define a decorator for deciding whether to execute
a function"""

import sys

from package_common.common_types import Any, Callable, Final
from package_common.default_logger import DefaultLogger


def yes_exe_no_exit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to execute a function when the character 'yes' is input
    and to exit the script when the character 'no' is input.

    Parameters
    ----------
    func : Callable
        The function executed when the character 'yes' is input.

    Returns
    ----------
    new_func : Callable
        A function executed when the character 'yes' is input.

    Warnings
    ----------
    Quit
        If the character 'n' or 'no' is input.
    Invalid input
        If characters other than 'y', 'yes', 'n', or 'no' are input.

    Examples
    ----------
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

    FUNCTION_NAME: Final[str] = sys._getframe().f_code.co_name

    def new_func(*args: tuple[Any, ...],
                 **kwargs: dict[str, Any]) -> Any:

        yes_no: str
        while True:
            yes_no = input('(yes/no) ').lower().strip()

            if yes_no in ('y', 'yes'):
                func(*args, **kwargs)
                break

            if yes_no in ('n', 'no'):
                DefaultLogger(FUNCTION_NAME).info('Quit')
                sys.exit(0)

            DefaultLogger(FUNCTION_NAME).error('Invalid input')
        #
    #

    return new_func
#


def exe_yes_continue(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to execute a function, and then to continue to execute
    a function while the character 'yes' is input.

    Parameters
    ----------
    func :
        The function executed when the character 'yes' is input.

    Returns
    ----------
    new_function : Callable
        The function executed when the character 'yes' is input.

    Warnings
    ----------
    Quit
        If the character 'n' or 'no' is input.
    Invalid input
        If characters other than 'y', 'yes', 'n', or 'no' are input.

    Examples
    ----------
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
    """

    FUNCTION_NAME: Final[str] = sys._getframe().f_code.co_name

    def new_function(*args: tuple[Any, ...],
                     **kwargs: dict[str, Any]) -> Any:

        yes_no: str = 'y'

        while True:
            func(*args, **kwargs)

            while True:
                yes_no = input('Re-execute? (yes/no) ').lower().strip()

                if yes_no in ('y', 'yes'):
                    break

                if yes_no in ('n', 'no'):
                    DefaultLogger(FUNCTION_NAME).info('Quit')
                    sys.exit(0)

                DefaultLogger(FUNCTION_NAME).error('Invalid input')

    return new_function
