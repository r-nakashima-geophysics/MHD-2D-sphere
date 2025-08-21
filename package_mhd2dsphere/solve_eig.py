"""A Python module to solve the eigenvalue problem of
two-dimensional (2D) magnetohydrodynamic (MHD) waves on a rotating
sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta).

References
----------
[1] Ryosuke Nakashima, Shigeo Yoshida, Two-dimensional ideal
magnetohydrodynamic waves on a rotating sphere under a non-Malkus field:
I. Continuous spectrum and its ray-theoretical interpretation.
Geophysical & Astrophysical Fluid Dynamics 118(5-6), 387-440 (2024).
doi: 10.1080/03091929.2024.2384388

[2] Ryosuke Nakashima (in prep.)
"""

import numpy as np

from package_common.background_field import BackgroundField
from package_common.common_types import (ArrayBool, ArrayComplex, ArrayFloat,
                                         ArrayStr)
from package_common.utils_debug import under_construction_log
from package_common.utils_eig import screening_eig, sort_eig
from package_mhd2dsphere.create_mat import create_mat, create_submat


def wrapper_solve_eig(
    m_order: int,
    alpha: float,
    e_eta: float,
    size_submat: int,
    *,
    criterion_c: dict[str, int | float],
    background_field: dict[str, BackgroundField | bool]) \
        -> tuple[ArrayComplex,
                 ArrayComplex,
                 ArrayComplex,
                 ArrayFloat,
                 ArrayFloat,
                 ArrayFloat,
                 ArrayStr]:
    """Solve the eigenvalue problem for a given alpha.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    alpha : float
        The Lehnert number.
    e_eta : float
        The magnetic Ekman number.
    size_submat : int
        The size of submatrices.
    criterion_c : dict[str, int | float]
        The criterion for convergence.
    background_field : dict[str, BackgroundField | bool]
        The background field.

    Returns
    -------
    psi_vec : ArrayComplex
        The eigenvector for the stream function.
    vpa_vec : ArrayComplex
        The eigenvector for the vector potential.
    eig : ArrayComplex
        The eigenvalue.
    mke : ArrayFloat
        The mean kinetic energy.
    mme : ArrayFloat
        The mean magnetic energy.
    ohm : ArrayFloat
        The Ohmic dissipation.
    sym : ArrayStr
        The symmetry of the eigenmodes.
    """

    size_mat: int = 2 * size_submat

    submatrices: tuple[ArrayFloat,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayFloat] \
        = create_submat(m_order, size_submat,
                        background_field=background_field)

    mat: ArrayFloat | ArrayComplex = create_mat(
        m_order, alpha, e_eta, submatrices,
        background_field=background_field)

    eig_valvec: ArrayComplex
    phys_qtys: tuple[ArrayFloat,
                     ArrayFloat,
                     ArrayFloat,
                     ArrayStr]
    eig_valvec, phys_qtys = solve_eig(m_order, alpha, e_eta, mat,
                                      criterion_c=criterion_c,
                                      background_field=background_field)

    psi_vec: ArrayComplex = eig_valvec[:size_submat, :]
    vpa_vec: ArrayComplex = eig_valvec[size_submat:size_mat, :]

    eig: ArrayComplex = eig_valvec[size_mat, :]

    return psi_vec, vpa_vec, eig, *phys_qtys


def solve_eig(m_order: int,
              alpha: float,
              e_eta: float,
              mat: ArrayFloat | ArrayComplex,
              *,
              criterion_c: dict[str, int | float],
              background_field: dict[str, BackgroundField | bool]) \
        -> tuple[ArrayComplex,
                 tuple[ArrayFloat,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayStr]]:
    """Solves the eigenvalue problem.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    alpha : float
        The Lehnert number.
    e_eta : float
        The magnetic Ekman number.
    mat: ArrayComplex
        The total matrix.
    criterion_c : dict[str, int | float]
        The criterion for convergence.
    background_field : dict[str, BackgroundField | bool]
        The background field.

    Returns
    -------
    eig_valvec : ArrayComplex
        The eigenvalues and normalized eigenvectors.
    phys_qtys : tuple[ArrayFloat, ArrayFloat, ArrayFloat, ArrayStr]
        The mean kinetic energy, mean magnetic energy, Ohmic
        dissipation, and symmetry of the eigenmodes.
    """

    eig_val: ArrayComplex
    eig_vec: ArrayComplex
    eig_val, eig_vec = np.linalg.eig(mat)

    eig_valvec: ArrayComplex = sort_eig(eig_val, eig_vec)
    eig_valvec = normalize_eigvec(m_order, eig_valvec,
                                  background_field=background_field)
    phys_qtys: tuple[ArrayFloat,
                     ArrayFloat,
                     ArrayFloat,
                     ArrayStr] = calc_qty(m_order, e_eta, eig_valvec,
                                          background_field=background_field)
    check: ArrayBool \
        = check_eig(m_order, alpha, eig_valvec, criterion_c=criterion_c)
    eig_valvec, phys_qtys = screening_eig(eig_valvec, check, *phys_qtys)

    return eig_valvec, phys_qtys


def normalize_eigvec(
    m_order: int,
    eig_valvec: ArrayComplex,
    *,
    background_field: dict[str, BackgroundField | bool]) \
        -> ArrayComplex:
    """Normalize the eigenvectors.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.

    Returns
    -------
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    """

    size_mat: int = eig_valvec.shape[1]

    mke: ArrayFloat
    mme: ArrayFloat
    mke, mme = calc_ene(m_order, eig_valvec,
                        background_field=background_field)
    eig_valvec[0*size_mat:1*size_mat, :] /= np.sqrt(mke+mme)

    return eig_valvec


def calc_qty(m_order: int,
             e_eta: float,
             eig_valvec: ArrayComplex,
             *,
             background_field: dict[str, BackgroundField | bool]) \
    -> tuple[ArrayFloat,
             ArrayFloat,
             ArrayFloat,
             ArrayStr]:
    """Calculate various physical quantities from the eigenvectors.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    e_eta : float
        The magnetic Ekman number.
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    background_field : dict[str, BackgroundField | bool]
        The background field.

    Returns
    -------
    mke : ArrayFloat
        The mean kinetic energy.
    mme : ArrayFloat
        The mean magnetic energy.
    ohm : ArrayFloat
        The ohmic dissipation.
    sym : ArrayStr
        The symmetry of eigenmodes.
    """

    mke: ArrayFloat
    mme: ArrayFloat
    mke, mme = calc_ene(m_order, eig_valvec,
                        background_field=background_field)

    if background_field['NY24']:

        size_mat: int = eig_valvec.shape[1]
        size_submat: int = int(size_mat/2)

        ohm: ArrayFloat = np.zeros(size_mat)
        if e_eta != 0:
            n_degree: int
            nn1: int
            for i_n in range(size_submat):
                n_degree = m_order + i_n
                nn1 = n_degree * (n_degree+1)

                ohm += (nn1**2) * (
                    np.abs(eig_valvec[size_submat+i_n, :])**2)

            ohm *= e_eta

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

    else:
        under_construction_log()

    return mke, mme, ohm, sym


def calc_ene(m_order: int,
             eig_valvec: ArrayComplex,
             *,
             background_field: dict[str, BackgroundField | bool]) \
    -> tuple[ArrayFloat,
             ArrayFloat]:
    """Calculate the mean kinetic and magnetic energies.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    background_field : dict[str, BackgroundField | bool]
        The background field.

    Returns
    -------
    mke : ArrayFloat
        The mean kinetic energy.
    mme : ArrayFloat
        The mean magnetic energy.

    Notes
    -----
    This function is based on eq. (24) in Nakashima & Yoshida
    (2024)[1]_.
    """

    if background_field['NY24']:

        size_mat: int = eig_valvec.shape[1]
        size_submat: int = int(size_mat/2)

        mke: ArrayFloat = np.zeros(size_mat, dtype=np.float64)
        mme: ArrayFloat = np.zeros(size_mat, dtype=np.float64)

        n_degree: int
        nn1: int
        for i_n in range(size_submat):
            n_degree = m_order + i_n
            nn1 = n_degree * (n_degree+1)

            mke += nn1 * (np.abs(eig_valvec[i_n, :])**2)
            mme += nn1 * (np.abs(eig_valvec[size_submat+i_n, :])**2)

    else:
        under_construction_log()

    return mke, mme


def check_eig(m_order: int,
              alpha: float,
              eig_valvec: ArrayComplex,
              *,
              criterion_c: dict[str, int | float]) -> ArrayBool:
    """Check the validity of eigenmodes.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    alpha : float
        The Lehnert number.
    eig_valvec : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    criterion_c : dict[str, int | float]
        A criterion for convergence.

    Returns
    -------
    check : ArrayBool
        The validity of eigenmodes.

    Notes
    -----
    This function is based on eq. (23) in Nakashima & Yoshida
    (2024)[1]_.
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
        check = (low_psi > high_psi*r_c) * (low_a > high_a*r_c)
    else:
        check = low_psi > high_psi*r_c

    return check
