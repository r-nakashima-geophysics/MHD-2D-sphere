"""A Python module to define a class for handling a log message."""

import logging


class DefaultLogger:
    """Class to handle a log message.

    Attributes
    ----------
    __logger : logging.Logger
        The instance of the logger.

    Examples
    --------
    >>> from package.default_logger import DefaultLogger
    >>> logger = DefaultLogger(name="my_logger")
    >>> logger.debug("This is a debug message.")
    >>> logger.info("This is an info message.")
    >>> logger.warning("This is a warning message.")
    >>> logger.error("This is an error message.")
    >>> logger.critical("This is a critical message.")
    """

    def __init__(self,
                 name: str,
                 level: int = logging.DEBUG) -> None:
        """Initializer for DefaultLogger class.

        Parameters
        ----------
        name : str
            The name of the logger.
        level : int, optional
            The logging level, default logging.DEBUG.
        """
        self.__logger: logging.Logger = logging.getLogger(name)
        self.__logger.setLevel(level)
        self.__logger.propagate = False

        if not self.__logger.handlers:
            fmt: str \
                = "\033[1m/%(levelname)s/\033[0m" \
                + " [%(asctime)s] %(name)s: %(message)s"
            handler: logging.StreamHandler = logging.StreamHandler()
            formatter: logging.Formatter = logging.Formatter(
                fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)
            self.__logger.addHandler(handler)

    def debug(self,
              message: str) -> None:
        """Method to log a debug message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.__logger.debug(message)

    def info(self,
             message: str) -> None:
        """Method to log an information message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.__logger.info(message)

    def warning(self,
                message: str) -> None:
        """Method to log a warning message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.__logger.warning(message)

    def error(self,
              message: str) -> None:
        """Method to log an error message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.__logger.error(message)

    def critical(self,
                 message: str) -> None:
        """Method to log a critical message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.__logger.critical(message)
