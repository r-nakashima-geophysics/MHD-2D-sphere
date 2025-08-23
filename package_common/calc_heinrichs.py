"""A Python module to calculate values related to the Heinrichs basis,
(1-x**2)T_n(x), where T_n(x) is the Chebyshev polynomial of degree n.

References
----------
[1] John P. Boyd, Chebyshev and Fourier Spectral Methods. Courier
Corporation, (2001).
"""

from package_common.calc_chebyshev import (chebyshev, chebyshev_d,
                                           chebyshev_d2, chebyshev_d3)
from package_common.common_types import TypeVarFloatComplex


def heinrichs(n_degree: int,
              s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the Heinrichs basis at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the Heinrichs basis at the point.

    Examples
    --------
    >>> from package_common.calc_heinrichs import heinrichs
    >>> print(heinrichs(3, 0.5))
    -0.75
    """

    return (1-(s_pos**2)) * chebyshev(n_degree, s_pos)


def heinrichs_d(n_degree: int,
                s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the first derivative of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the first derivative of the Heinrichs basis at the
        point.

    Examples
    --------
    >>> from package_common.calc_heinrichs import heinrichs_d
    >>> print(heinrichs_d(3, 0.5))
    1.0000000000000002
    """

    return (
        (1-(s_pos**2)) * chebyshev_d(n_degree, s_pos)
        - 2 * s_pos * chebyshev(n_degree, s_pos)
    )


def heinrichs_d2(n_degree: int,
                 s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the second derivative of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the second derivative of the Heinrichs basis at the
        point.

    Examples
    --------
    >>> from package_common.calc_heinrichs import heinrichs_d2
    >>> print(heinrichs_d2(3, 0.5))
    11.000000000000002
    """

    return (
        (1-(s_pos**2)) * chebyshev_d2(n_degree, s_pos)
        - 4 * s_pos * chebyshev_d(n_degree, s_pos)
        - 2 * chebyshev(n_degree, s_pos)
    )


def heinrichs_d3(n_degree: int,
                 s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the third derivative of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the third derivative of the Heinrichs basis at the
        point.

    Examples
    --------
    >>> from package_common.calc_heinrichs import heinrichs_d3
    >>> print(heinrichs_d3(3, 0.5))
    -18.000000000000004
    """

    return (
        (1-(s_pos**2)) * chebyshev_d3(n_degree, s_pos)
        - 6 * s_pos * chebyshev_d2(n_degree, s_pos)
        - 6 * chebyshev_d(n_degree, s_pos)
    )
