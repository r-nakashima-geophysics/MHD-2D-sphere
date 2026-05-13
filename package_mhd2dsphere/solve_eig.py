"""A Python module to solve the eigenvalue problem of
two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and background zonal flows, U_phi = U_0 U(theta) sin(theta).

References
----------
[1] Ryosuke Nakashima, Shigeo Yoshida, Two-dimensional ideal
magnetohydrodynamic waves on a rotating sphere under a non-Malkus field: I.
Continuous spectrum and its ray-theoretical interpretation. Geophysical &
Astrophysical Fluid Dynamics 118(5-6), 387-440 (2024). doi:
10.1080/03091929.2024.2384388

[2] Ryosuke Nakashima, Shigeo Yoshida (in prep.)
"""

import numpy as np

from package_common.calc_heinrichs import heinrichs
from package_common.common_types import (ArrayBool, ArrayComplex, ArrayFloat,
                                         ArrayStr)
from package_common.spectral_deform import ComplexCoordinate
from package_common.utils_collocation import (chebyshev_gauss_quad,
                                              spherical_laplacian_heinrichs)
from package_common.utils_debug import under_construction_log
from package_common.utils_eig import screening_eig, sort_eig
from package_mhd2dsphere.create_mat import create_mat, create_submat
from package_mhd2dsphere.typed_dict import (DictBackgroundField,
                                            DictCriterionC, DictPhysQtys,
                                            DictResult)


def wrapper_solve_eig(
        m_order: int,
        alpha: float,
        e_eta: float,
        rossby: float,
        size_submat: int,
        *,
        criterion_c: DictCriterionC,
        background_field: DictBackgroundField) -> DictResult:
    """Solve the eigenvalue problem for a given alpha.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    alpha : float
        The Lehnert number.
    e_eta : float
        The magnetic Ekman number.
    rossby : float
        The Rossby number.
    size_submat : int
        The size of submatrices.
    criterion_c : DictCriterionC
        The criterion for convergence.
    background_field : DictBackgroundField
        The background field.

    Returns
    -------
    DictResult
        The dictionary of the result of the eigenvalue problem.
    """

    submatrices: tuple[ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex] \
        = create_submat(m_order, e_eta, rossby, size_submat,
                        background_field=background_field)

    mat: ArrayFloat | ArrayComplex = create_mat(
        m_order, alpha, e_eta, submatrices,
        background_field=background_field)

    return solve_eig(m_order, alpha, e_eta, mat,
                     criterion_c=criterion_c,
                     background_field=background_field)


def solve_eig(m_order: int,
              alpha: float,
              e_eta: float,
              mat: ArrayFloat | ArrayComplex,
              *,
              criterion_c: DictCriterionC,
              background_field: DictBackgroundField) -> DictResult:
    """Solves the eigenvalue problem.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    alpha : float
        The Lehnert number.
    e_eta : float
        The magnetic Ekman number.
    mat: ArrayFloat | ArrayComplex
        The total matrix.
    criterion_c : DictCriterionC
        The criterion for convergence.
    background_field : DictBackgroundField
        The background field.

    Returns
    -------
    result : DictResult
        The dictionary of the result of the eigenvalue problem.
    """

    eig_val: ArrayComplex
    eig_vec: ArrayComplex
    eig_val, eig_vec = np.linalg.eig(mat)

    eig_valvec: ArrayComplex = sort_eig(eig_val, eig_vec)
    phys_qtys: DictPhysQtys = calc_qty(m_order, e_eta, eig_valvec,
                                       background_field=background_field)
    check: ArrayBool \
        = check_eig(m_order, alpha, eig_valvec, criterion_c=criterion_c)

    tuple_phys_qtys: tuple[ArrayFloat,
                           ArrayFloat,
                           ArrayFloat,
                           ArrayFloat,
                           ArrayFloat,
                           ArrayStr]
    eig_valvec, tuple_phys_qtys \
        = screening_eig(eig_valvec, check, *phys_qtys.values())
    for key, value in zip(phys_qtys.keys(), tuple_phys_qtys):
        phys_qtys[key] = value

    size_mat: int = eig_valvec.shape[1]
    size_submat: int = int(size_mat/2)
    result: DictResult = {
        'lin_alpha': None,
        'eig': eig_valvec[size_mat, :],
        'vec_psi': eig_valvec[:size_submat, :],
        'vec_vpa': eig_valvec[size_submat:size_mat, :],
        'phys_qtys': phys_qtys
    }

    return result


def calc_qty(m_order: int,
             e_eta: float,
             eig_valvec: ArrayComplex,
             *,
             background_field: DictBackgroundField) -> DictPhysQtys:
    """Calculate various physical quantities from the eigenvectors.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    e_eta : float
        The magnetic Ekman number.
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    background_field : DictBackgroundField
        The background field.

    Returns
    -------
    phys_qtys : DictPhysQtys
        The dictionary of the physical quantities.
    """

    size_mat: int = eig_valvec.shape[1]
    size_submat: int = int(size_mat/2)

    pke: ArrayFloat
    pme: ArrayFloat
    eig_valvec, pke, pme = normalize_eigvec(m_order, eig_valvec,
                                            background_field=background_field)

    psm: ArrayFloat = np.zeros(size_mat)
    pse: ArrayFloat = np.zeros(size_mat)

    ohm: ArrayFloat = np.zeros(size_mat)
    if e_eta != 0:
        if background_field['NY24']:
            n_degree: int
            for i_n in range(size_submat):
                n_degree = m_order + i_n

                ohm += (n_degree**2) * ((n_degree+1)**2) * (
                    np.abs(eig_valvec[size_submat+i_n, :])**2)
            ohm *= e_eta
        else:
            under_construction_log()

    even: ArrayFloat = np.zeros(size_mat)
    odd: ArrayFloat = np.zeros(size_mat)
    sym: ArrayStr = np.empty(size_mat, dtype=np.str_)
    for i_n in range(int(size_submat/2)):
        even += np.abs(eig_valvec[2*i_n, :])
        odd += np.abs(eig_valvec[2*i_n+1, :])

    for i_mode in range(size_mat):
        if even[i_mode] > odd[i_mode]:
            sym[i_mode] = 'sinuous'
        else:
            sym[i_mode] = 'varicose'

    phys_qtys: DictPhysQtys = {
        'pke': pke,
        'pme': pme,
        'psm': psm,
        'pse': pse,
        'ohm': ohm,
        'sym': sym
    }

    return phys_qtys


def normalize_eigvec(
    m_order: int,
    eig_valvec: ArrayComplex,
    *,
    background_field: DictBackgroundField) \
        -> tuple[ArrayComplex, ArrayFloat, ArrayFloat]:
    """Normalize the eigenvectors.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    background_field : DictBackgroundField
        The background field.

    Returns
    -------
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    pke : ArrayFloat
        The perturbation kinetic energy.
    pme : ArrayFloat
        The perturbation magnetic energy.
    """

    size_mat: int = eig_valvec.shape[1]

    pke: ArrayFloat
    pme: ArrayFloat
    pke, pme = calc_ene(m_order, eig_valvec,
                        background_field=background_field)

    total_energy: ArrayFloat = pke + pme

    eig_valvec[0*size_mat:1*size_mat, :] /= np.sqrt(total_energy)

    pke /= total_energy
    pme /= total_energy

    return eig_valvec, pke, pme


def calc_ene(m_order: int,
             eig_valvec: ArrayComplex,
             *,
             background_field: DictBackgroundField) \
    -> tuple[ArrayFloat,
             ArrayFloat]:
    """Calculate the perturbation kinetic and magnetic energies.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    background_field : DictBackgroundField
        The background field.

    Returns
    -------
    pke : ArrayFloat
        The perturbation kinetic energy.
    pme : ArrayFloat
        The perturbation magnetic energy.

    Notes
    -----
    This function is based on eq. (24) in Nakashima & Yoshida (2024)[1]_. When
    SWITCH_NY24 is False, the Chebyshev-Gauss quadrature is used.
    """

    size_mat: int = eig_valvec.shape[1]
    size_submat: int = int(size_mat/2)

    vec_psi: ArrayComplex = eig_valvec[:size_submat, :]
    vec_vpa: ArrayComplex = eig_valvec[size_submat:size_mat, :]

    pke: ArrayFloat = np.zeros(size_mat, dtype=np.float64)
    pme: ArrayFloat = np.zeros(size_mat, dtype=np.float64)

    if background_field['NY24']:
        n_degree: int
        nn1: int
        for i_n in range(size_submat):
            n_degree = m_order + i_n
            nn1 = n_degree * (n_degree+1)

            pke += nn1 * (np.abs(vec_psi[i_n, :])**2)
            pme += nn1 * (np.abs(vec_vpa[i_n, :])**2)
    else:

        mu_complex: ComplexCoordinate = background_field['MU']

        def _minus_spherical_laplacian_heinrichs(
                n_degree: int,
                s_pos: float | complex) -> float | complex:
            return (-1) * spherical_laplacian_heinrichs(
                m_order, n_degree, s_pos, mu_complex=mu_complex)

        pke = np.real(chebyshev_gauss_quad(
            size_mat,
            vec_1=vec_psi, func_1=heinrichs,
            vec_2=vec_psi, func_2=_minus_spherical_laplacian_heinrichs,
            y_complex=mu_complex
        ))
        pme = np.real(chebyshev_gauss_quad(
            size_mat,
            vec_1=vec_vpa, func_1=heinrichs,
            vec_2=vec_vpa, func_2=_minus_spherical_laplacian_heinrichs,
            y_complex=mu_complex
        ))

    return pke, pme


def check_eig(m_order: int,
              alpha: float,
              eig_valvec: ArrayComplex,
              *,
              criterion_c: DictCriterionC) -> ArrayBool:
    """Check the validity of eigenmodes.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    alpha : float
        The Lehnert number.
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    criterion_c : DictCriterionC
        A criterion for convergence.

    Returns
    -------
    check : ArrayBool
        The validity of eigenmodes.

    Notes
    -----
    This function is based on eq. (23) in Nakashima & Yoshida (2024)[1]_.
    """

    n_c: int = int(criterion_c['degree'])
    r_c: float = criterion_c['ratio']

    size_mat: int = eig_valvec.shape[1]
    size_submat: int = int(size_mat/2)

    n_degree: int

    low_psi: ArrayFloat = np.zeros(size_mat, dtype=np.float64)
    low_a: ArrayFloat = np.zeros(size_mat, dtype=np.float64)
    high_psi: ArrayFloat = np.zeros(size_mat, dtype=np.float64)
    high_a: ArrayFloat = np.zeros(size_mat, dtype=np.float64)
    for i_n in range(size_submat):
        n_degree = m_order + i_n

        if n_degree <= n_c:
            low_psi += (np.abs(eig_valvec[i_n, :])**2)
            low_a += (np.abs(eig_valvec[size_submat+i_n, :])**2)
        else:
            high_psi += (np.abs(eig_valvec[i_n, :])**2)
            high_a += (np.abs(eig_valvec[size_submat+i_n, :])**2)

    check: ArrayBool = np.empty(size_mat, dtype=np.bool_)
    if alpha != 0:
        check = (low_psi > high_psi*r_c) & (low_a > high_a*r_c)
    else:
        check = low_psi > high_psi*r_c

    return check
