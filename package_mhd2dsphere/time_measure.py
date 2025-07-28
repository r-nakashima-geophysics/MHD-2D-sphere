"""Measures computational times"""

from time import perf_counter


def time_progress(num_calc: int,
                  i_calc: int,
                  previous_time: float) -> float:
    """Measure calculation times and displays its progress

    Parameters
    ----------
    num_cal : int
        The total iteration number
    i_calc : int
        The current iteration number
    previous_time : float
        The previous timestamp

    Returns
    ----------
    now
        The current timestamp

    """

    now: float = perf_counter()
    difference: float = now - previous_time
    finish_min: float = ((num_calc-i_calc-1)*difference) / 60

    print(f'({i_calc+1}/{num_calc}) {difference:.3f} s '
          + f'[finish: {finish_min:.1f} minutes after]')

    return now
#
