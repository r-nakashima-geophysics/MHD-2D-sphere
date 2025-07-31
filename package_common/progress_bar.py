"""A Python module to display the progress bar"""

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
    """

    num_mark: int = 40
    mark_blank: str = ' '
    mark_filled: str = '█'

    rap_time: Optional[float] = timer.rap()

    if name is None:
        name = ''

    p_bar: str
    text: str

    if rap_time is not None:
        rate: int = int(((i_calc+1)/num_calc) * num_mark)
        remaining_hours: float \
            = (num_calc - i_calc - 1) * rap_time / 3600
        p_bar = mark_filled * rate + mark_blank * (num_mark - rate)
        text = f'Finish {remaining_hours:.1f} hours later'
        print(
            f'\r{name} [{p_bar}] {i_calc+1}/{num_calc}: {text}', end='')

    if i_calc == num_calc - 1:
        p_bar = mark_filled * num_mark
        len_text: int = len(text)
        text = 'Finished'
        blank_text: str = ' ' * (len_text - len(text))
        text += blank_text
        print(f'\r{name} [{p_bar}] {i_calc+1}/{num_calc}: {text}')
