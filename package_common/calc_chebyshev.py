"""A Python module to calculate values related to Chebyshev
polynomials.

References
----------
[1] John P. Boyd, Chebyshev and Fourier Spectral Methods. Courier
Corporation, (2001).
"""

import numpy as np

from package_common.common_types import TypeVarFloatComplex


def chebyshev(n_degree: int,
              s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of a Chebyshev polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the Chebyshev polynomial at the point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev
    >>> print(chebyshev(3, 0.5))
    -1.0
    """

    return np.cos(n_degree * np.acos(s_pos))


def chebyshev_d(n_degree: int,
                s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the first derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the first derivative of the Chebyshev polynomial at
        the point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev_d
    >>> print(chebyshev_d(3, 0.5))
    4.2423009548996277e-16
    """

    t: TypeVarFloatComplex = np.acos(s_pos)
    return n_degree * np.sin(n_degree*t) / np.sin(t)


def chebyshev_d2(n_degree: int,
                 s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the second derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the second derivative of the Chebyshev polynomial
        at the point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev_d2
    >>> print(chebyshev_d2(3, 0.5))
    12.000000000000002
    """

    t: TypeVarFloatComplex = np.acos(s_pos)
    return (
        (-(n_degree**2) * np.cos(n_degree*t)
            + chebyshev_d(n_degree, s_pos) * np.cos(t)
         ) / (np.sin(t)**2)
    )


def chebyshev_d3(n_degree: int,
                 s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the third derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the third derivative of the Chebyshev polynomial at
        the point.

    Examples
    ----------
    >>> from package_common.calc_chebyshev import chebyshev_d3
    >>> print(chebyshev_d3(3, 0.5))
    24.000000000000007
    """

    t: TypeVarFloatComplex = np.acos(s_pos)
    return (
        ((1-(n_degree**2)) * chebyshev_d(n_degree, s_pos)
            + 3 * chebyshev_d2(n_degree, s_pos) * np.cos(t)
         ) / (np.sin(t)**2)
    )
