"""A Python module to calculate values related to Chebyshev
polynomials."""

import numpy as np
from numba import njit

from package_common.common_types import TypeVar

T = TypeVar('T', float, complex)


@njit
def chebyshev(n_degree: int,
              s_pos: T) -> T:
    """Calculate the value of a Chebyshev polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : T
        The position of the point.

    Returns
    ----------
    T
        The value of the Chebyshev polynomial at the point.
    """

    return np.cos(n_degree * np.acos(s_pos))


@njit
def chebyshev_d(n_degree: int,
                s_pos: T) -> T:
    """Calculate the value of the first derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : T
        The position of the point.

    Returns
    ----------
    T
        The value of the first derivative of the Chebyshev polynomial at
        the point.
    """

    t: T = np.acos(s_pos)
    return n_degree * np.sin(n_degree*t) / np.sin(t)


@njit
def chebyshev_d2(n_degree: int,
                 s_pos: T) -> T:
    """Calculate the value of the second derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : T
        The position of the point.

    Returns
    ----------
    T
        The value of the second derivative of the Chebyshev polynomial
        at the point.
    """

    t: T = np.acos(s_pos)
    return (-n_degree**2 * np.cos(n_degree*t)
            + chebyshev_d(n_degree, s_pos) * np.cos(t)
            ) / (np.sin(t)**2)


@njit
def chebyshev_d3(n_degree: int,
                 s_pos: T) -> T:
    """Calculate the value of the third derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : T
        The position of the point.

    Returns
    ----------
    T
        The value of the third derivative of the Chebyshev polynomial at
        the point.

    """

    t: T = np.acos(s_pos)
    return ((1-n_degree**2) * chebyshev_d(n_degree, s_pos)
            + chebyshev_d2(n_degree, s_pos) * 3 * np.cos(t)
            ) / (np.sin(t)**2)
