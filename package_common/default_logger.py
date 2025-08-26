"""A Python module to define a class for handling log messages."""

import logging
import sys


class DefaultLogger:
    """Class to handle log messages.

    Examples
    --------
    >>> from package_common.default_logger import DefaultLogger
    >>> logger = DefaultLogger('my_logger')
    >>> logger.debug('This is a debug message.')
    >>> logger.info('This is an info message.')
    >>> logger.warning('This is a warning message.')
    >>> logger.error('This is an error message.')
    >>> logger.critical('This is a critical message.')
    >>> param1 = 1
    >>> param2 = 2
    >>> logger.show_params(f'{param1=}', f'{param2=}')
    """

    def __init__(self,
                 name: str,
                 level: int | str = logging.INFO) -> None:
        """Initialize an instance of the DefaultLogger class.

        Parameters
        ----------
        name : str
            The name of the logger.
        level : int, optional, default logging.INFO
            The logging level.

        Warnings
        --------
        Invalid input
            If the input string in the (optional) second argument is not
            either 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
        """

        if isinstance(level, str):
            try:
                level = getattr(logging, level.upper())
            except AttributeError:
                print(f'/ERROR/ {self.__class__.__name__}'
                      + ': Invalid input')
                sys.exit(1)

        self.__logger: logging.Logger = logging.getLogger(name)
        self.__logger.setLevel(level)
        self.__logger.propagate = False

        if not self.__logger.handlers:
            fmt: str = \
                '/%(levelname)s/ [%(asctime)s] %(name)s: %(message)s'
            handler: logging.StreamHandler = logging.StreamHandler()
            handler.setLevel(level)
            formatter: logging.Formatter = logging.Formatter(
                fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.__logger.addHandler(handler)

    def debug(self,
              message: str) -> None:
        """Log a debug message.

        Parameters
        ----------
        message : str
            The message to log.
        """

        self.__logger.debug(message)

    def info(self,
             message: str) -> None:
        """Log an information message.

        Parameters
        ----------
        message : str
            The message to log.
        """

        self.__logger.info(message)

    def warning(self,
                message: str) -> None:
        """Log a warning message.

        Parameters
        ----------
        message : str
            The message to log.
        """

        self.__logger.warning(message)

    def error(self,
              message: str) -> None:
        """Log an error message.

        Parameters
        ----------
        message : str
            The message to log.
        """

        self.__logger.error(message)

    def critical(self,
                 message: str) -> None:
        """Log a critical message.

        Parameters
        ----------
        message : str
            The message to log.
        """

        self.__logger.critical(message)

    def show_params(self,
                    *args) -> None:
        """Show the parameters.

        Parameters
        ----------
        *args
            The parameters to log.
        """

        self.__logger.info('----- Parameters -----')
        for parameter in args:
            parameter = parameter.replace('=', ' = ')
            self.__logger.info(parameter)
        self.__logger.info('----------------------')
