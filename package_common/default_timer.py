"""A Python module to define a class for measuring the computational
time of a Python script.

Note
----
The caffeine module will be imported in the initializer of the
DefaultTimer class when this class is used on macOS.
"""

import importlib
import sys
from time import perf_counter
from types import ModuleType

from package_common.default_logger import DefaultLogger


class DefaultTimer:
    """Class to measure the computational time of a Python script.

    Examples
    --------
    >>> from package_common.default_timer import DefaultTimer
    >>> timer = DefaultTimer('my_timer')
    >>> timer.start()
    >>> timer.show()
    >>> _ = timer.lap()  # None
    >>> lap_time = timer.lap()
    >>> timer.end()
    """

    __import_caffeine: bool = False

    def __init__(self,
                 name: str) -> None:
        """Initialize an instance of the DefaultTimer class.

        Parameters
        ----------
        name : str
            The name of the timer.
        """

        self.__start_time: float | None = None
        self.__elapsed_time: float | None = None
        self.__split_time: float | None = None

        self.__logger: DefaultLogger = DefaultLogger(name)

        if (sys.platform == "darwin") \
                and (not DefaultTimer.__import_caffeine):

            caffeine: ModuleType = importlib.import_module('caffeine')
            caffeine.on(display=False)
            DefaultTimer.__import_caffeine = True

    def start(self) -> None:
        """Start the timer."""

        self.__elapsed_time = None
        self.__split_time = None

        self.__logger.info('Start')
        self.__start_time = perf_counter()

    def show(self) -> None:
        """Show the elapsed time.

        Warnings
        --------
        Timer has not been started.
            If `start()` has not been called before `show()` or `end()`
            are called.
        """

        if self.__start_time is None:
            self.__logger.warning('Timer has not been started.')
        else:
            self.__elapsed_time = perf_counter() - self.__start_time
            self.__logger.info(
                f'Elapsed time: {self.__elapsed_time:.1f} sec.')

    def end(self) -> None:
        """End the timer."""

        self.show()
        self.__logger.info('End')

    def lap(self) -> float | None:
        """Measure the lap time.

        Returns
        -------
        float | None
            The lap time.
        """

        now: float = perf_counter()
        if self.__split_time is None:
            self.__split_time = now
            return None
        lap_time: float = now - self.__split_time
        self.__split_time = now
        return lap_time
