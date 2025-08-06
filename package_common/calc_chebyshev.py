"""A Python module to calculate values related to Chebyshev
polynomials"""

import cmath

from numba import njit


@njit
def chebyshev(n_degree: int,
              s_pos: complex) -> complex:
    """Calculate the value of a Chebyshev polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial
    s_pos : complex
        The position of the point

    Returns
    ----------
    complex
        The value of the Chebyshev polynomial at the point
    """

    return cmath.cos(n_degree * cmath.acos(s_pos))


@njit
def chebyshev_d(n_degree: int,
                s_pos: complex) -> complex:
    """Calculate the value of the first derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial
    s_pos : complex
        The position of the point

    Returns
    ----------
    complex
        The value of the first derivative of the Chebyshev polynomial at
        the point
    """

    t: complex = cmath.acos(s_pos)
    return n_degree * cmath.sin(n_degree*t) / cmath.sin(t)


@njit
def chebyshev_d2(n_degree: int,
                 s_pos: complex) -> complex:
    """Calculate the value of the second derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial
    s_pos : complex
        The position of the point

    Returns
    ----------
    complex
        The value of the second derivative of the Chebyshev polynomial
        at the point
    """

    t: complex = cmath.acos(s_pos)
    return (-n_degree**2 * cmath.cos(n_degree*t)
            + chebyshev_d(n_degree, s_pos) * cmath.cos(t)
            ) / (cmath.sin(t)**2)


@njit
def chebyshev_d3(n_degree: int,
                 s_pos: complex) -> complex:
    """Calculate the value of the third derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial
    s_pos : complex
        The position of the point

    Returns
    ----------
    complex
        The value of the third derivative of the Chebyshev polynomial at
        the point

    """

    t: complex = cmath.acos(s_pos)
    return ((1-n_degree**2) * chebyshev_d(n_degree, s_pos)
            + chebyshev_d2(n_degree, s_pos) * 3 * cmath.cos(t)
            ) / (cmath.sin(t)**2)
