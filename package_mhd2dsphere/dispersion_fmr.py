"""Defines the approximate dispersion relation for fast magnetic Rossby
(MR) waves under the non-Malkus field B_phi = B_0 sin(theta) cos(theta)

"""

import math

import numpy as np
from scipy.integrate import quad


def dispersion_fmr(eig: float,
                   *args) -> float:
    """The approximate dispersion relation for fast magnetic Rossby (MR)
    waves under the non-Malkus field B_phi = B_0 sin(theta) cos(theta)

    Parameters
    ----------
    eig : float
        An eigenvalue
    args : tuple of int and float and str
        A tuple of parameters other than eig
        (m_order, n_degree, alpha, switch_eq)

    Returns
    ----------
    dispersion_relation : float
        If dispersion_relation = 0, the dispersion relation is satisfied
        for given parameters.

    Notes
    ----------
    This function is based on eqs. (B.2) and (F.10) in Nakashima &
    Yoshida (submitted)[1]_.

    """

    m_order: int
    n_degree: int
    alpha: float
    switch_eq: str
    m_order, n_degree, alpha, switch_eq = args

    ma2: float = (m_order**2) * (alpha**2)

    dispersion_relation: float = math.nan
    if switch_eq == 'sph':
        dispersion_relation = -(m_order/eig) + ma2/(eig**2) \
            - n_degree * (n_degree+1) - (1/2)*(
            1-(2*m_order-1)*(2*m_order+1)
                / ((2*n_degree-1)*(2*n_degree+3))) \
            * (ma2/(eig**2))*(7+m_order/eig)
    elif switch_eq == 'wkb':
        kappa2: float = (-(eig/m_order)-ma2) / ((eig**2)-ma2)
        m_eig: float = m_order * eig
        uc_n: int = n_degree - m_order

        def integrand(var: float,
                      kappa2: float,
                      m_eig: float) -> float:
            func: float = 1 / (
                ((var**2)+kappa2)
                * np.sqrt(((var**2)+1)*((var**2)-m_eig*kappa2)))
            return func
        #

        integral, _ = quad(
            integrand, 0, math.inf, args=(kappa2, m_eig))
        dispersion_relation = 2*(1+m_eig) \
            * np.sqrt(-m_eig/((eig**2)-ma2)) \
            * kappa2*integral - (uc_n+(1/2))*math.pi
    #

    return dispersion_relation
#
