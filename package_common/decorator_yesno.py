"""A Python module to define a decorator for deciding whether to execute
a function or not."""

import sys

from package_common.common_types import Callable
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


def yes_exe_no_quit(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator to execute a function when the character 'yes' is input
    and to exit the script when the character 'no' is input.

    Parameters
    ----------
    func : Callable[..., None]
        The function executed when the character 'yes' is input.

    Returns
    -------
    new_func : Callable[..., None]
        The wrapped function.

    Warnings
    --------
    Quit
        If the character 'n' or 'no' is input.
    Invalid input
        If characters other than 'y', 'yes', 'n', or 'no' are input.

    Examples
    --------
    >>> from package_common.decorator_yesno import yes_exe_no_quit
    >>> def test():
    ...     print('test')
    ...
    >>> @yes_exe_no_quit
    ... def wrapper():
    ...     test()
    ...
    >>> wrapper()
    Enter yes or no: yes
    test
    """

    logger: DefaultLogger = create_function_name_logger()

    def new_func(*args,
                 **kwargs) -> None:

        yes_or_no: str
        while True:
            yes_or_no \
                = input('Enter \'yes\' or \'no\': ').strip().lower()

            if yes_or_no in ('y', 'yes'):
                func(*args, **kwargs)
                break

            if yes_or_no in ('n', 'no'):
                logger.info('Quit')
                sys.exit(0)

            logger.error('Invalid input')

    return new_func


def exe_yes_continue(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator to execute a function, and then to continue to execute
    it while the character 'yes' is input.

    Parameters
    ----------
    func : Callable[..., None]
        The function executed when the character 'yes' is input.

    Returns
    -------
    new_func : Callable[..., None]
        The wrapped function.

    Warnings
    --------
    Quit
        If the character 'n' or 'no' is input.
    Invalid input
        If characters other than 'y', 'yes', 'n', or 'no' are input.

    Examples
    --------
    >>> from package_common.decorator_yesno import exe_yes_continue
    >>> def test():
    ...     print('test')
    ...
    >>> @exe_yes_continue
    ... def wrapper():
    ...     test()
    ...
    >>> wrapper()
    test
    Re-execute? Enter yes or no: yes
    test
    Re-execute? Enter yes or no: no
    """

    logger: DefaultLogger = create_function_name_logger()

    def new_func(*args,
                 **kwargs) -> None:

        yes_or_no: str
        while True:
            func(*args, **kwargs)

            while True:
                yes_or_no = input(
                    'Re-execute? Enter \'yes\' or \'no\': '
                ).strip().lower()

                if yes_or_no in ('y', 'yes'):
                    break

                if yes_or_no in ('n', 'no'):
                    logger.info('Quit')
                    sys.exit(0)

                logger.error('Invalid input')

    return new_func
