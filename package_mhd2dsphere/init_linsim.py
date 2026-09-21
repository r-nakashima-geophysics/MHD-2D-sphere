"""A Python module to construct the initial values of Field class for the
(quasi-)linear simulation of two-dimensional (2D) incompressible
magnetohydrodynamic (MHD) waves on a rotating sphere under a toroidal
background field, B_phi = B_0 B(theta) sin(theta), and a background zonal flow,
U_phi = U_0 U(theta) sin(theta)."""

import numpy as np
from scipy.special import assoc_legendre_p, legendre_p

from package_common.common_types import ArrayFloat
from package_common.default_logger import DefaultLogger
from package_common.utils_collocation import (calc_collocation_point,
                                              create_cheb_expan_mat)
from package_common.utils_name import create_function_name_logger


def init_spherical_harmonics(size_submat: int,
                             *,
                             n_degree: int,
                             m_order: int) -> ArrayFloat:
    """Compute the Heinrichs expansion coefficients of the spherical harmonics.

    Parameters
    ----------
    size_submat : int
        The size of submatrices.
    n_degree : int
        The degree of the spherical harmonics.
    m_order : int
        The order of the spherical harmonics.

    Returns
    -------
    ArrayFloat
        The Heinrichs expansion coefficients of the spherical harmonics.

    Warnings
    --------
    Not supported
        If `m_order` is 1 or 0.
    Invalid argument
        If `size_submat` is not positive.

    Examples
    --------
    >>> from package_mhd2dsphere.init_linsim import init_spherical_harmonics
    >>> init_spherical_harmonics(5, n_degree=2, m_order=2)
    array([ 3.00000000e+00, -4.99600361e-16,  0.00000000e+00, -1.16573418e-15,
        6.66133815e-16])
    """

    if m_order <= 1:
        logger: DefaultLogger = create_function_name_logger()
        logger.error('Not supported')

    if (size_submat > 0) and (n_degree >= m_order):

        mu: ArrayFloat = np.array(
            [calc_collocation_point(i_l+1, size_submat+2)
             for i_l in range(size_submat)])
        pnm: ArrayFloat = np.squeeze(assoc_legendre_p(n_degree, m_order, mu))
        expan_mat: ArrayFloat = create_cheb_expan_mat(size_submat+2)

        if m_order >= 3:
            coeff: ArrayFloat = expan_mat @ np.concatenate(
                ([0], pnm/(1-(mu**2)), [0])
            )
        elif m_order == 2:
            coeff: ArrayFloat = expan_mat @ np.concatenate(
                ([legendre_p(n_degree, -1, diff_n=2)[2]],
                 pnm/(1-(mu**2)),
                 [legendre_p(n_degree, 1, diff_n=2)[2]])
            )

        return coeff[:size_submat]

    logger: DefaultLogger = create_function_name_logger()
    logger.error('Invalid argument')
