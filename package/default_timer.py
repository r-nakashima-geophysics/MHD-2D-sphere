"""A Python module to define a class for measuring the computational
time of a Python script.
"""

from time import perf_counter
from typing import Optional

from package.default_logger import DefaultLogger


class DefaultTimer:
    """Class to measure the computational time of a Python script.

    Attributes
    ----------
    __logger : DefaultLogger
        The instance of the logger.
    __start_time : float
        The starting time of the timer.
    __elapsed_time : float
        The elapsed time.

    Examples
    --------
    >>> from package.default_timer import DefaultTimer
    >>> timer = DefaultTimer(name="my_timer")
    >>> timer.start()
    >>> timer.end()
    """

    def __init__(self, name: str) -> None:
        """Initializer for DefaultTimer class.

        Parameters
        ----------
        name : str
            The name of the timer.
        """

        self.__logger: DefaultLogger = DefaultLogger(name=name)
        self.__start_time: Optional[float] = None
        self.__elapsed_time: Optional[float] = None

    def start(self) -> None:
        """Instance method to start the timer."""
        self.__logger.info("Start")
        self.__start_time = perf_counter()

    def show(self) -> None:
        """Instance method to show the elapsed time."""
        if self.__start_time is None:
            self.__logger.warning("Timer has not been started.")
        else:
            self.__elapsed_time = perf_counter() - self.__start_time
            self.__logger.info(
                f"Elapsed time: {self.__elapsed_time:.2f} sec.")

    def end(self) -> None:
        """Instance method to end the timer."""
        self.show()
        self.__logger.info("End")
