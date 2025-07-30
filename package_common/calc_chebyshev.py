"""A Python module to calculate the values relating to the Chebyshev
polynomials"""

import cmath
import math

from numba import njit

from package_common.common_types import TypeVar

T = TypeVar("T", float, complex)


@njit
def chebyshev(n_degree: int,
              s_pos: T) -> T:
    """Calculate the value of a Chebyshev polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial
    s_pos : T
        The position of the point

    Returns
    ----------
    value_chebyshev : T
        The value of the Chebyshev polynomial at the point
    """

    if isinstance(s_pos, float):
        return math.cos(n_degree * math.acos(s_pos))

    return cmath.cos(n_degree * cmath.acos(s_pos))


@njit
def chebyshev_d(n_degree: int,
                s_pos: T) -> T:
    """Calculate the value of the first derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial
    s_pos : T
        The position of the point

    Returns
    ----------
    value_chebyshev_d : T
        The value of the first derivative of the Chebyshev polynomial at
        the point
    """

    value_chebyshev_d: T = 0.0
    if n_degree == 0:
        return value_chebyshev_d

    if n_degree % 2 == 0:
        for i_n in range(1, n_degree, 2):
            value_chebyshev_d += chebyshev(i_n, s_pos)
    else:
        for i_n in range(2, n_degree, 2):
            value_chebyshev_d += chebyshev(i_n, s_pos)
        value_chebyshev_d += chebyshev(0, s_pos) / 2

    value_chebyshev_d *= 2 * n_degree

    return value_chebyshev_d


@njit
def chebyshev_d2(n_degree: int,
                 s_pos: T) -> T:
    """Calculate the value of the second derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial
    s_pos : T
        The position of the point

    Returns
    ----------
    value_chebyshev_d2 : T
        The value of the second derivative of the Chebyshev polynomial
        at the point
    """

    value_chebyshev_d2: T = 0.0
    if 0 <= n_degree <= 1:
        return value_chebyshev_d2

    if n_degree % 2 == 0:
        for i_n in range(2, n_degree-1, 2):
            value_chebyshev_d2 += ((n_degree**2)-(i_n**2)) \
                * chebyshev(i_n, s_pos)
        value_chebyshev_d2 \
            += (n_degree**2) * chebyshev(0, s_pos) / 2
    else:
        for i_n in range(1, n_degree-1, 2):
            value_chebyshev_d2 += ((n_degree**2)-(i_n**2)) \
                * chebyshev(i_n, s_pos)

    value_chebyshev_d2 *= n_degree

    return value_chebyshev_d2


@njit
def chebyshev_d3(n_degree: int,
                 s_pos: T) -> T:
    """Calculate the value of the third derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial
    s_pos : T
        The position of the point

    Returns
    ----------
    value_chebyshev_d3 : T
        The value of the third derivative of the Chebyshev polynomial at
        the point

    """

    value_chebyshev_d3: T = 0.0
    if 0 <= n_degree <= 2:
        return value_chebyshev_d3

    if n_degree % 2 == 0:
        for i_n in range(1, n_degree-2, 2):
            value_chebyshev_d3 += (((n_degree+1)**2)-(i_n**2)) \
                * (((n_degree-1)**2)-(i_n**2)) \
                * chebyshev(i_n, s_pos)
    else:
        for i_n in range(2, n_degree-2, 2):
            value_chebyshev_d3 += (((n_degree+1)**2)-(i_n**2)) \
                * (((n_degree-1)**2)-(i_n**2)) \
                * chebyshev(i_n, s_pos)
        value_chebyshev_d3 \
            += ((n_degree+1)**2) * ((n_degree-1)**2) \
            * chebyshev(0, s_pos) / 2

    value_chebyshev_d3 *= n_degree / 4

    return value_chebyshev_d3
