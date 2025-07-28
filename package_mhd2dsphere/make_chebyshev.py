"""Makes data of Chebyshev polynomials"""

import cmath

from numba import njit


@njit
def chebyshev(n_degree: int,
              s_complex: complex) -> complex:
    """Calculates the value of a Chebyshev polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of a Chebyshev polynomial
    s_complex : complex
        The position of a point

    Returns
    ----------
    value_chebyshev : complex
        The value of a Chebyshev polynomial at a point

    """

    value_chebyshev: complex = cmath.cos(n_degree*cmath.acos(s_complex))

    return value_chebyshev
#


@njit
def chebyshev_d(n_degree: int,
                s_complex: complex) -> complex:
    """Calculates the value of the first derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of a Chebyshev polynomial
    s_complex : complex
        The position of a point

    Returns
    ----------
    value_chebyshev_d : complex
        The value of the first derivative of a Chebyshev polynomial at
        a point

    """

    value_chebyshev_d: complex = 0
    if n_degree == 0:
        value_chebyshev_d = 0
    elif n_degree % 2 == 0:
        for i_n in range(1, n_degree, 2):
            value_chebyshev_d += chebyshev(i_n, s_complex)
    else:
        for i_n in range(2, n_degree, 2):
            value_chebyshev_d += chebyshev(i_n, s_complex)
        value_chebyshev_d += chebyshev(0, s_complex) / 2
    #

    value_chebyshev_d *= 2 * n_degree

    return value_chebyshev_d
#


@njit
def chebyshev_d2(n_degree: int,
                 s_complex: complex) -> complex:
    """Calculates the value of the second derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of a Chebyshev polynomial
    s_complex : complex
        The position of a point

    Returns
    ----------
    value_chebyshev_d2 : complex
        The value of the second derivative of a Chebyshev polynomial at
        a point

    """

    value_chebyshev_d2: complex = 0
    if 0 <= n_degree <= 1:
        value_chebyshev_d2 = 0
    elif n_degree % 2 == 0:
        for i_n in range(2, n_degree-1, 2):
            value_chebyshev_d2 += ((n_degree**2)-(i_n**2)) \
                * chebyshev(i_n, s_complex)
        value_chebyshev_d2 \
            += (n_degree**2) * chebyshev(0, s_complex) / 2
    else:
        for i_n in range(1, n_degree-1, 2):
            value_chebyshev_d2 += ((n_degree**2)-(i_n**2)) \
                * chebyshev(i_n, s_complex)
    #

    value_chebyshev_d2 *= n_degree

    return value_chebyshev_d2
#


@njit
def chebyshev_d3(n_degree: int,
                 s_complex: complex) -> complex:
    """Calculates the value of the third derivative of a Chebyshev
    polynomial at a point

    Parameters
    ----------
    n_degree : int
        The degree of a Chebyshev polynomial
    s_complex : complex
        The position of a point

    Returns
    ----------
    value_chebyshev_d3 : complex
        The value of the third derivative of a Chebyshev polynomial at
        a point

    """

    value_chebyshev_d3: complex = 0
    if 0 <= n_degree <= 2:
        value_chebyshev_d3 = 0
    elif n_degree % 2 == 0:
        for i_n in range(1, n_degree-2, 2):
            value_chebyshev_d3 += (((n_degree+1)**2)-(i_n**2)) \
                * (((n_degree-1)**2)-(i_n**2)) \
                * chebyshev(i_n, s_complex)
    else:
        for i_n in range(2, n_degree-2, 2):
            value_chebyshev_d3 += (((n_degree+1)**2)-(i_n**2)) \
                * (((n_degree-1)**2)-(i_n**2)) \
                * chebyshev(i_n, s_complex)
        value_chebyshev_d3 \
            += ((n_degree+1)**2) * ((n_degree-1)**2) \
            * chebyshev(0, s_complex) / 2
    #

    value_chebyshev_d3 *= n_degree / 4

    return value_chebyshev_d3
#
