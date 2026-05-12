"""A Python module to create the matrices for the eigenvalue problem of
two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and background zonal flows, U_phi = U_0 U(theta) sin(theta).

References
----------
[1] Ryosuke Nakashima, Shigeo Yoshida, Two-dimensional ideal
magnetohydrodynamic waves on a rotating sphere under a non-Malkus field:
I. Continuous spectrum and its ray-theoretical interpretation.
Geophysical & Astrophysical Fluid Dynamics 118(5-6), 387-440 (2024).
doi: 10.1080/03091929.2024.2384388

[2] Ryosuke Nakashima (in prep.)
"""

import sys

import numpy as np

from package_common.background_field import BackgroundField
from package_common.calc_heinrichs import heinrichs, heinrichs_d, heinrichs_d2
from package_common.common_types import ArrayComplex, ArrayFloat
from package_common.default_logger import DefaultLogger
from package_common.spectral_deform import ComplexCoordinate
from package_common.utils_name import create_function_name_logger
from package_mhd2dsphere.typed_dict import DictBackgroundField


def create_submat(m_order: int,
                  e_eta: float,
                  rossby: float,
                  size_submat: int,
                  *,
                  background_field: DictBackgroundField) \
    -> tuple[ArrayFloat | ArrayComplex,
             ArrayFloat | ArrayComplex,
             ArrayFloat | ArrayComplex,
             ArrayFloat | ArrayComplex]:
    """Create the submatrices.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    e_eta : float
        The magnetic Ekman number.
    rossby : float
        The Rossby number.
    size_submat : int
        The size of submatrices.
    background_field : DictBackgroundField
        The background field.

    Returns
    -------
    submat_11 : ArrayFloat | ArrayComplex
        The (1,1)th block matrix.
    submat_12 : ArrayFloat | ArrayComplex
        The (1,2)th block matrix.
    submat_21 : ArrayFloat | ArrayComplex
        The (2,1)th block matrix.
    submat_22 : ArrayFloat | ArrayComplex
        The (2,2)th block matrix.

    Notes
    -----
    This function is based on eq. (22) in Nakashima & Yoshida
    (2024)[1]_.
    """

    submat_11: ArrayFloat | ArrayComplex
    submat_12: ArrayFloat | ArrayComplex
    submat_21: ArrayFloat | ArrayComplex
    submat_22: ArrayFloat | ArrayComplex

    n_t: int
    if background_field['NY24']:
        n_t = size_submat + m_order - 1
        lin_n: ArrayFloat = np.linspace(
            m_order, n_t, size_submat, dtype=np.float64)

        knm: ArrayFloat = np.sqrt((lin_n-m_order) * (lin_n+m_order)
                                  / ((2*lin_n-1)*(2*lin_n+1)))

        submat_11 = np.zeros(
            (size_submat, size_submat), dtype=np.float64)
        for i_submat in range(size_submat):
            submat_11[i_submat, i_submat] \
                = 1 / ((m_order+i_submat)*(m_order+1+i_submat))

        submat_12 = np.zeros(
            (size_submat, size_submat), dtype=np.float64)
        for i_submat in range(size_submat-1):
            submat_12[i_submat, i_submat+1] \
                = (m_order-1+i_submat) * (m_order+4+i_submat) \
                * knm[1+i_submat] \
                / ((m_order+i_submat)*(m_order+1+i_submat))
            submat_12[i_submat+1, i_submat] \
                = (m_order-2+i_submat) * (m_order+3+i_submat) \
                * knm[1+i_submat] \
                / ((m_order+1+i_submat)*(m_order+2+i_submat))

        submat_21 = np.zeros(
            (size_submat, size_submat), dtype=np.float64)
        for i_submat in range(size_submat-1):
            submat_21[i_submat, i_submat+1] = knm[1+i_submat]
            submat_21[i_submat+1, i_submat] = knm[1+i_submat]

        submat_22 = np.zeros(
            (size_submat, size_submat), dtype=np.float64)
        for i_submat in range(size_submat):
            submat_22[i_submat, i_submat] \
                = (m_order+i_submat) * (m_order+1+i_submat)

    else:
        n_t = size_submat - 1

        bg_field_b: BackgroundField = background_field['B']
        bg_field_u: BackgroundField = background_field['U']
        mu_complex: ComplexCoordinate = background_field['MU']

        if mu_complex.check_spectral_deform() or (e_eta != 0):
            submat_11 = np.zeros(
                (size_submat, size_submat), dtype=np.complex128)
            submat_12 = np.zeros(
                (size_submat, size_submat), dtype=np.complex128)
            submat_21 = np.zeros(
                (size_submat, size_submat), dtype=np.complex128)
            submat_22 = np.zeros(
                (size_submat, size_submat), dtype=np.complex128)
            submat_b_11 = np.zeros(
                (size_submat, size_submat), dtype=np.complex128)
            submat_b_22 = np.zeros(
                (size_submat, size_submat), dtype=np.complex128)
        else:
            submat_11 = np.zeros(
                (size_submat, size_submat), dtype=np.float64)
            submat_12 = np.zeros(
                (size_submat, size_submat), dtype=np.float64)
            submat_21 = np.zeros(
                (size_submat, size_submat), dtype=np.float64)
            submat_22 = np.zeros(
                (size_submat, size_submat), dtype=np.float64)
            submat_b_11 = np.zeros(
                (size_submat, size_submat), dtype=np.float64)
            submat_b_22 = np.zeros(
                (size_submat, size_submat), dtype=np.float64)

        s_pos: float
        mu: float | complex
        u_mu: float | complex
        b_mu: float | complex
        u_shear_mu: float | complex
        b_shear_mu: float | complex
        for i_l in range(size_submat):
            s_pos = calc_collocation_point(i_l+1, n_t+2)

            if mu_complex.check_spectral_deform() or (e_eta != 0):
                mu = mu_complex.value(s_pos)
                u_mu = bg_field_u.value(mu)
                u_shear_mu = (
                    bg_field_u.value_d2(mu) * (1-(mu**2))
                    - 4 * mu * bg_field_u.value_d(mu)
                    - 2 * bg_field_u.value(mu)
                )
                b_mu = bg_field_b.value(mu)
                b_shear_mu = (
                    bg_field_b.value_d2(mu) * (1-(mu**2))
                    - 4 * mu * bg_field_b.value_d(mu)
                    - 2 * bg_field_b.value(mu)
                )
            else:
                mu = mu_complex.r_value(s_pos)
                u_mu = bg_field_u.r_value(mu)
                u_shear_mu = (
                    bg_field_u.r_value_d2(mu) * (1-(mu**2))
                    - 4 * mu * bg_field_u.r_value_d(mu)
                    - 2 * bg_field_u.r_value(mu)
                )
                b_mu = bg_field_b.r_value(mu)
                b_shear_mu = (
                    bg_field_b.r_value_d2(mu) * (1-(mu**2))
                    - 4 * mu * bg_field_b.r_value_d(mu)
                    - 2 * bg_field_b.r_value(mu)
                )

            for i_n in range(size_submat):
                h_n: float = heinrichs(i_n, s_pos)
                laplacian: float | complex = laplacian_heinrichs(
                    m_order, i_n, s_pos,
                    background_field=background_field)

                submat_11[i_l, i_n] \
                    = rossby*u_mu*laplacian + h_n \
                    - rossby*u_shear_mu*h_n
                submat_12[i_l, i_n] = b_mu*laplacian - b_shear_mu*h_n
                submat_21[i_l, i_n] = b_mu * h_n
                submat_22[i_l, i_n] \
                    = m_order*rossby*u_mu*h_n + 1j*e_eta*laplacian

                submat_b_11[i_l, i_n] = laplacian
                submat_b_22[i_l, i_n] = h_n

        inv_submat_b_11: ArrayFloat | ArrayComplex
        inv_submat_b_22: ArrayFloat | ArrayComplex
        if mu_complex.check_spectral_deform() or (e_eta != 0):
            inv_submat_b_11 \
                = np.linalg.inv(submat_b_11).astype(np.complex128)
            inv_submat_b_22 \
                = np.linalg.inv(submat_b_22).astype(np.complex128)
        else:
            inv_submat_b_11 \
                = np.linalg.inv(submat_b_11).astype(np.float64)
            inv_submat_b_22 \
                = np.linalg.inv(submat_b_22).astype(np.float64)

        submat_11 = inv_submat_b_11 @ submat_11
        submat_12 = inv_submat_b_11 @ submat_12
        submat_21 = inv_submat_b_22 @ submat_21
        submat_22 = inv_submat_b_22 @ submat_22

    return submat_11, submat_12, submat_21, submat_22


def calc_collocation_point(i_l: int,
                           num_point: int) -> float:
    """Calculate the Gauss-Lobatto collocation points.

    Parameters
    ----------
    i_l : int
        The index of the collocation point.
    num_point : int
        The number of the collocation points.

    Returns
    -------
    float
        The position of the collocation point.

    Warnings
    --------
    Invalid input
        If the input value is not within [0, num_point].
    """

    logger: DefaultLogger = create_function_name_logger()
    if 0 <= i_l <= num_point:
        return -np.cos(i_l*np.pi/num_point)

    logger.error('Invalid input')
    sys.exit(1)


def laplacian_heinrichs(
        m_order: int,
        n_degree: int,
        s_pos: float,
        *,
        background_field: DictBackgroundField) -> float | complex:
    """Calculate the spherical horizontal Laplacian of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : float
        The position of the point.
    background_field : DictBackgroundField
        The background field.

    Returns
    -------
    TypeVarFloatComplex
        The value of the spherical horizontal Laplacian of the Heinrichs
        basis at the point.
    """

    mu_complex: ComplexCoordinate = background_field['MU']

    mu: float | complex
    mu_d: float | complex
    mu_d2: float | complex
    if mu_complex.check_spectral_deform():
        mu = mu_complex.value(s_pos)
        mu_d = mu_complex.value_d(s_pos)
        mu_d2 = mu_complex.value_d2(s_pos)
    else:
        mu = mu_complex.r_value(s_pos)
        mu_d = mu_complex.r_value_d(s_pos)
        mu_d2 = mu_complex.r_value_d2(s_pos)

    sin2: float | complex = 1 - (mu**2)

    return (
        sin2 * heinrichs_d2(n_degree, s_pos) / (mu_d**2)
        - (2*mu/mu_d + sin2*mu_d2/(mu_d**3))
        * heinrichs_d(n_degree, s_pos)
        - (m_order**2) * heinrichs(n_degree, s_pos) / sin2
    )


def create_mat(m_order: int,
               alpha: float,
               e_eta: float,
               submatrices: tuple[ArrayFloat | ArrayComplex,
                                  ArrayFloat | ArrayComplex,
                                  ArrayFloat | ArrayComplex,
                                  ArrayFloat | ArrayComplex],
               *,
               background_field: DictBackgroundField) \
        -> ArrayFloat | ArrayComplex:
    """Make the total matrix.

    Parameters
    ----------
    m_order: int
        The zonal wavenumber(order).
    alpha: float
        The Lehnert number.
    e_eta: float
        The magnetic Ekman number.
    submatrices: tuple[ArrayFloat | ArrayComplex, ArrayFloat |
    ArrayComplex, ArrayFloat | ArrayComplex, ArrayFloat | ArrayComplex]
        The(1, 1)th, (1, 2)th, (2, 1)th, and (2, 2)th block matrices.
    background_field: DictBackgroundField
        The background field.

    Returns
    -------
    mat: ArrayFloat | ArrayComplex
        The total matrix.

    Notes
    -----
    This function is based on eq. (22) in Nakashima & Yoshida
    (2024)[1]_.
    """

    submat_11: ArrayFloat | ArrayComplex
    submat_12: ArrayFloat | ArrayComplex
    submat_21: ArrayFloat | ArrayComplex
    submat_22: ArrayFloat | ArrayComplex
    submat_11, submat_12, submat_21, submat_22 = submatrices

    size_submat: int = submat_11.shape[0]
    size_mat: int = 2 * size_submat

    if background_field['NY24']:

        if e_eta == 0:
            mat = np.zeros((size_mat, size_mat), dtype=np.float64)
        else:
            mat = np.zeros((size_mat, size_mat), dtype=np.complex128)

        mat[0*size_submat:1*size_submat,
            0*size_submat:1*size_submat] = -m_order * submat_11
        mat[0*size_submat:1*size_submat,
            1*size_submat:2*size_submat] = -m_order * alpha * submat_12

        mat[1*size_submat:2*size_submat,
            0*size_submat:1*size_submat] = -m_order * alpha * submat_21
        if e_eta != 0:
            mat[1*size_submat:2*size_submat,
                1*size_submat:2*size_submat] = -1j * e_eta * submat_22

    else:
        mu_complex: ComplexCoordinate = background_field['MU']

        if mu_complex.check_spectral_deform() or (e_eta != 0):
            mat = np.zeros((size_mat, size_mat), dtype=np.complex128)
        else:
            mat = np.zeros((size_mat, size_mat), dtype=np.float64)

        mat[0*size_submat:1*size_submat,
            0*size_submat:1*size_submat] = m_order * submat_11
        mat[0*size_submat:1*size_submat,
            1*size_submat:2*size_submat] = -m_order * alpha * submat_12

        mat[1*size_submat:2*size_submat,
            0*size_submat:1*size_submat] = -m_order * alpha * submat_21
        mat[1*size_submat:2*size_submat,
            1*size_submat:2*size_submat] = submat_22

    return mat
