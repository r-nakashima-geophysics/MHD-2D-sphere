"""Makes the Frobenius solutions of 2D ideal MHD waves on a rotating
sphere under the non-Malkus field B_phi = B_0 sin(theta) cos(theta)

References
----------
[1] Nakashima & Yoshida (submitted)

"""

import math

import numpy as np


def calc_frobenius(m_order: int,
                   alpha: float,
                   num_theta: int,
                   eig: complex,
                   mu_c: complex) -> tuple[np.ndarray, np.ndarray]:
    """Calculates some coefficients of the Frobenius solutions for
    the non-Malkus field B_phi = B_0 sin(theta) cos(theta)

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order)
    alpha : float
        The Lehnert number
    num_theta : int
        The number of the grid in the theta direction
    eig : complex
        An eigenvalue
    mu_c : complex
        A critical latitude

    Returns
    ----------
    psi1 : ndarray
        The first Frobenius solution
    psi2 : ndarray
        The second Frobenius solution

    Notes
    ----------
    This function is based on eqs. (17) and (18) in Nakashima & Yoshida
    (submitted)[1]_.

    """

    bc2_d: list[complex] = [complex(), ] * 4
    bc2_d[0] = mu_c ** 2
    bc2_d[1] = 2 * mu_c
    bc2_d[2] = 2
    bc2_d[3] = 0

    sin2: complex = 1 - (mu_c**2)
    denominator: complex = bc2_d[1] * sin2

    const_d: list[complex] = [complex(), ] * 4
    const_d[0] = -(eig/(m_order*(alpha**2))+bc2_d[1]*mu_c+2*bc2_d[0]) \
        / denominator
    const_d[1] = (2*bc2_d[1]*mu_c-bc2_d[2]*sin2/2) / denominator
    const_d[2] \
        = ((m_order**2)*bc2_d[1]/sin2-bc2_d[2]*mu_c-3*bc2_d[1]) \
        / denominator
    const_d[3] = (bc2_d[1]+bc2_d[2]*mu_c-bc2_d[3]*sin2/6) / denominator

    coef_a: list[complex] = [complex(), ] * 4
    coef_b: list[complex] = [complex(), ] * 4
    coef_a[0] = const_d[0]
    coef_a[1] \
        = ((const_d[0]**2)+2*const_d[0]*const_d[1]+const_d[2]) / 4
    coef_b[0] = -2*const_d[0] + const_d[1]
    coef_b[1] = (-3*(const_d[0]**2)-2*const_d[0]*const_d[1] +
                 2*(const_d[1]**2)-const_d[2]+2*const_d[3]) / 4

    lin_theta: np.ndarray = np.linspace(0, math.pi, num_theta)
    delta_mu: np.ndarray = np.cos(lin_theta) - np.full(num_theta, mu_c)

    psi1 = 1 + coef_a[0]*delta_mu + coef_a[1]*(delta_mu**2)
    psi2 = psi1*np.log(delta_mu) \
        + coef_b[0]*delta_mu + coef_b[1]*(delta_mu**2)

    return psi1, psi2
#


def make_fitting_data(psi1: np.ndarray,
                      psi2: np.ndarray,
                      num_data: int,
                      i_theta_c: int,
                      eq_or_pole: str) -> np.ndarray:
    """Makes data for the fittings of the Frobenius solutions

    Parameters
    ----------
    psi1 : ndarray
        An eigenfunction of the stream function (psi)
    psi2 : ndarray
        An eigenfunction of the stream function (psi)
    num_data : int
        The number of data points to which we will fit
    i_theta_c : int
        The index of the discrete position of a critical latitude
    eq_or_pole : str
        'eq' or 'pole'

    """

    psi1_data: np.ndarray = np.full(num_data, np.nan)
    psi2_data: np.ndarray = np.full(num_data, np.nan)
    data: np.ndarray = np.full(num_data, np.nan)

    if eq_or_pole == 'eq':
        psi1_data = psi1[i_theta_c+1:i_theta_c+num_data+1].real
        psi2_data = psi2[i_theta_c+1:i_theta_c+num_data+1].real
    elif eq_or_pole == 'pole':
        psi1_data = psi1[i_theta_c-num_data:i_theta_c].real
        psi2_data = psi2[i_theta_c-num_data:i_theta_c].real
    #

    data = psi1_data / psi2_data

    return data
#
