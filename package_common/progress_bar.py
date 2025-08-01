"""A Python module to display the progress bar."""

from package_common.common_types import Optional
from package_common.default_timer import DefaultTimer


def progress_bar(num_calc: int,
                 i_calc: int,
                 timer: DefaultTimer,
                 name: Optional[str] = None) -> None:
    """Measure calculation times and display the progress bar.

    Parameters
    ----------
    num_calc : int
        The total iteration number.
    i_calc : int
        The current iteration number.
    timer : DefaultTimer
        The instance of the timer.
    name : Optional[str], optional, default None
        The name of the progress bar.

    Examples
    --------
    >>> timer = DefaultTimer("my_timer")
    >>> n = 100
    >>> for i in range(n):
    ...     progress_bar(n, i, timer, "my_progress_bar")
    """

    bar_width: int = 20
    mark_empty: str = ' '
    mark_filled: str = '█'

    rap_time: Optional[float] = timer.rap()

    if name is None:
        name = ''

    p_bar: str
    text: str

    if rap_time is not None:
        remaining_hours: float = (num_calc-i_calc-1) * rap_time / 3600
        len_filled: int = int(((i_calc+1)/num_calc) * bar_width)
        p_bar = mark_filled * len_filled \
            + mark_empty * (bar_width - len_filled)
        text = f'{i_calc+1}/{num_calc}: ' \
            + f'Finish {remaining_hours:.1f} hrs later ' \
            + f'(rap: {rap_time:.2f} sec)'
        print(f'\r{name} [{p_bar}] {text}', end='')

    if i_calc == num_calc - 1:
        p_bar = mark_filled * bar_width
        len_text: int = len(text)
        text = f'{i_calc+1}/{num_calc}: Finished'
        print(f'\r{name} [{p_bar}] {text:<{len_text}}')
