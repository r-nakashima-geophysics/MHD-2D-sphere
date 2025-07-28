"""A Python module to define a class for measuring the computational
time of a Python script.
"""

from time import perf_counter

from package_common.common_types import Optional
from package_common.default_logger import DefaultLogger


class DefaultTimer:
    """Class to measure the computational time of a Python script.

    Attributes
    ----------
    __logger : DefaultLogger
        The instance of the logger.
    __start_time : Optional[float]
        The starting time of the timer.
    __elapsed_time : Optional[float]
        The elapsed time.

    Warnings
    ----------
    Timer has not been started.
        If `start()` has not been called before `show()` or `end()` are
        called.

    Examples
    --------
    >>> my_timer = DefaultTimer(name="my_timer")
    >>> my_timer.start()
    >>> my_timer.end()
    """

    def __init__(self,
                 name: str) -> None:
        """Initialize the DefaultTimer instance.

        Parameters
        ----------
        name : str
            The name of the timer.
        """

        self.__logger: DefaultLogger = DefaultLogger(name=name)
        self.__start_time: Optional[float] = None
        self.__elapsed_time: Optional[float] = None

    def start(self) -> None:
        """Start the timer."""
        self.__logger.info("Start")
        self.__start_time = perf_counter()

    def show(self) -> None:
        """Show the elapsed time."""
        if self.__start_time is None:
            DefaultLogger(__class__.__name__).warning(
                "Timer has not been started.")
        else:
            self.__elapsed_time = perf_counter() - self.__start_time
            self.__logger.info(
                f"Elapsed time: {self.__elapsed_time:.1f} sec.")

    def end(self) -> None:
        """End the timer."""
        self.show()
        self.__logger.info("End")
