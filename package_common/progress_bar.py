"""A Python module to define a class for displaying the progress bar."""

import sys

from package_common.default_logger import DefaultLogger
from package_common.default_timer import DefaultTimer


class ProgressBar:
    """Class to display the progress bar.

    Examples
    --------
    >>> from package_common.progress_bar import ProgressBar
    >>> n = 100
    >>> progress_bar = ProgressBar('my_progress_bar', n)
    >>> for i in range(n):
    ...     if i == 0:
    ...         progress_bar.start()
    ...     else:
    ...         progress_bar.update(i)
    """

    __bar_width: int = 10
    __mark_empty: str = ' '
    __mark_filled: str = '█'
    __max_length_print_name: int = 15

    def __init__(self,
                 name: str,
                 num_calc: int) -> None:
        """Initialize an instance of the ProgressBar class.

        Parameters
        ----------
        name : str
            The name of the progress bar.
        num_calc : int
            The total iteration number.

        Warnings
        --------
        Invalid argument
            If the arguments are invalid.
        """

        self.__name: str = name
        self.__num_calc: int = num_calc

        self.__print_name: str = self.__name
        if len(self.__name) > ProgressBar.__max_length_print_name:
            self.__print_name \
                = self.__name[:ProgressBar.__max_length_print_name] \
                + '...'

        self.__logger: DefaultLogger = DefaultLogger(self.__name)
        self.__timer: DefaultTimer = DefaultTimer(self.__name)

        if self.__num_calc <= 0:
            self.__logger.warning('Invalid argument')
            sys.exit(1)

    def start(self) -> None:
        """Start the progress bar."""

        _ = self.__timer.lap()
        self.__logger.info('Start')

        text: str = f'0/{self.__num_calc}'
        p_bar: str = ProgressBar.__mark_empty * ProgressBar.__bar_width
        print(f'{self.__print_name} [{p_bar}] {text}',
              end='', flush=True)

    def update(self,
               i_calc: int,
               num_process: int = 1) -> None:
        """Measure calculation times and update the progress bar.

        Parameters
        ----------
        i_calc : int
            The current iteration number.
        num_process : int, optional, default 1
            The number of processes.

        Warnings
        --------
        Invalid argument
            If the arguments are invalid.
        Progress bar has not been started.
            If `start()` has not been called before `update()` is called.
        """

        if (self.__num_calc <= 0) or (i_calc < 0) or (
                i_calc + 1 > self.__num_calc):
            self.__logger.error('Invalid argument')
            sys.exit(1)

        if ((i_calc+1) % num_process == 0) \
                or (i_calc + 1 == self.__num_calc):

            lap_time: float | None = self.__timer.lap()
            if lap_time is None:
                self.__logger.warning(
                    'Progress bar has not been started.')
                self.start()
            else:
                remaining_hours: float \
                    = (self.__num_calc-i_calc-1) * lap_time / 3600 \
                    / num_process
                len_filled: int = int(((i_calc+1)/self.__num_calc)
                                      * ProgressBar.__bar_width)
                p_bar = ProgressBar.__mark_filled * len_filled \
                    + ProgressBar.__mark_empty \
                    * (ProgressBar.__bar_width - len_filled)
                text: str = f'{i_calc+1}/{self.__num_calc}' \
                    + f': Finish {remaining_hours:.1f} hrs later ' \
                    + f'(lap: {lap_time / num_process:.1f} sec)'

                if i_calc + 1 < self.__num_calc:
                    print(f'\r{self.__print_name} [{p_bar}] {text}',
                          end='', flush=True)
                elif i_calc + 1 == self.__num_calc:
                    len_text: int = len(text)
                    text = f'{i_calc+1}/{self.__num_calc}: Finished'
                    print(
                        f'\r{self.__print_name} [{p_bar}] '
                        + f'{text:<{len_text}}', flush=True)
                    self.__logger.info('End')
