"""A Python module to define a class for handling log messages."""

import logging


class DefaultLogger:
    """Class to handle log messages.

    Attributes
    ----------
    __logger : logging.Logger
        The instance of the logger.

    Examples
    --------
    >>> logger = DefaultLogger("my_logger")
    >>> logger.debug("This is a debug message.")
    >>> logger.info("This is an info message.")
    >>> logger.warning("This is a warning message.")
    >>> logger.error("This is an error message.")
    >>> logger.critical("This is a critical message.")
    """

    def __init__(self,
                 name: str,
                 level: int | str = logging.DEBUG) -> None:
        """Initialize the DefaultLogger instance.

        Parameters
        ----------
        name : str
            The name of the logger.
        level : int, optional, default logging.DEBUG
            The logging level.
        """

        self.__logger: logging.Logger = logging.getLogger(name)
        self.__logger.setLevel(level)
        self.__logger.propagate = False

        if not self.__logger.handlers:
            fmt: str = \
                "/%(levelname)s/ [%(asctime)s] %(name)s: %(message)s"
            handler: logging.StreamHandler = logging.StreamHandler()
            handler.setLevel(level)
            formatter: logging.Formatter = logging.Formatter(
                fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
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
