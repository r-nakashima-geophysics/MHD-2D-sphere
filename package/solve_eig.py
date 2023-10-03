"""Solves the eigenvalue problem

References
-----
[1] Nakashima & Yoshida (submitted)

"""

import math

import numpy as np

from package.make_mat import make_mat, make_submat


def wrapper_solve_eig(m_order: int,
                      alpha: float,
                      e_eta: float,
                      size_submat: int,
                      criterion_c: tuple[int, float]) \
        -> tuple[np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray]:
    """A wrapper of functions to solve the eigenvalue problem

    Parameters
    -----
    m_order : int
        The zonal wavenumber (order)
    alpha : float
        The Lehnert number
    e_eta : float
        The magnetic Ekman number
    size_submat : int
        The size of submatrices
    criterion_c : tuple of int and float
        A criterion for convergence (degree, ratio)

    Returns
    -----
    bundle : tuple of ndarray
        A tuple of results

    See Also
    -----
    package.make_mat.make_submat
    package.make_mat.make_mat
    solve_eig

    """

    size_mat: int = 2 * size_submat

    submatrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] \
        = make_submat(m_order, size_submat)

    mat: np.ndarray = make_mat(m_order, e_eta, submatrices, alpha)

    eig_vecval: np.ndarray
    phys_qtys: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    eig_vecval, phys_qtys \
        = solve_eig(m_order, e_eta, criterion_c, alpha, mat)

    psi_vec: np.ndarray = eig_vecval[:size_submat, :]
    vpa_vec: np.ndarray = eig_vecval[size_submat:size_mat, :]

    eig: np.ndarray = eig_vecval[size_mat, :]

    bundle: tuple[np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray] \
        = (psi_vec, vpa_vec, eig,
           phys_qtys[0], phys_qtys[1], phys_qtys[2], phys_qtys[3])

    return bundle
#


def solve_eig(m_order: int,
              e_eta: float,
              criterion_c: tuple[int, float],
              alpha: float,
              mat: np.ndarray) \
        -> tuple[np.ndarray,
                 tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Solves the eigenvalue problem

    Parameters
    -----
    m_order : int
        The zonal wavenumber (order)
    e_eta : float
        The magnetic Ekman number
    criterion_c : tuple of int and float
        A criterion for convergence (degree, ratio)
    alpha : float
        The Lehnert number
    mat: ndarray
        The main matrix

    Returns
    -----
    eig_vecval : ndarray
        Eigenvalues & normalized eigenvectors
    phys_qtys : tuple of ndarray
        mke, mme, ohm, sym

    """

    eig_val: np.ndarray
    eig_vec: np.ndarray
    eig_val, eig_vec = np.linalg.eig(mat)

    eig_vecval: np.ndarray = arrange_eig(m_order, eig_val, eig_vec)
    phys_qtys: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] \
        = calc_qty(m_order, e_eta, eig_vecval)
    check: np.ndarray \
        = check_eig(m_order, criterion_c, alpha, eig_vecval)
    eig_vecval, phys_qtys \
        = screening_eig(eig_vecval, phys_qtys, check)

    return eig_vecval, phys_qtys
#


def arrange_eig(m_order: int,
                eig_val: np.ndarray,
                eig_vec: np.ndarray) -> np.ndarray:
    """Arranges eigenvalues and normalized eigenvectors

    Parameters
    -----
    m_order : int
        The zonal wavenumber (order)
    eig_val : ndarray
        Eigenvalues
    eig_vec : ndarray
        Eigenvectors

    Returns
    -----
    eig_vecval : ndarray
        Eigenvalues & normalized eigenvectors

    """

    size_mat: int = eig_val.shape[0]

    eig_tmp: np.ndarray = np.zeros((2*size_mat+2, size_mat))
    eig_tmp[0*size_mat:1*size_mat, :] = eig_vec.real
    eig_tmp[1*size_mat:2*size_mat, :] = eig_vec.imag
    eig_tmp[2*size_mat, :] = eig_val.real
    eig_tmp[2*size_mat+1, :] = eig_val.imag
    eig_sorted: list[np.ndarray] \
        = sorted(eig_tmp.T, key=lambda x: x[2*size_mat])
    eig_tmp = np.array(eig_sorted).T

    eig_vecval: np.ndarray = np.zeros(
        (size_mat+1, size_mat), dtype=np.complex128)
    eig_vecval[0*size_mat:1*size_mat, :] \
        = eig_tmp[0*size_mat:1*size_mat, :] \
        + 1j*eig_tmp[1*size_mat:2*size_mat, :]
    eig_vecval[size_mat, :] \
        = eig_tmp[2*size_mat, :] + 1j*eig_tmp[2*size_mat+1, :]

    mke: np.ndarray
    mme: np.ndarray
    mke, mme = calc_ene(m_order, eig_vecval)
    eig_vecval[0*size_mat:1*size_mat, :] /= np.sqrt(mke+mme)

    return eig_vecval
#


def calc_ene(m_order: int,
             eig_vecval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the mean kinetic and magnetic energies

    Parameters
    -----
    m_order : int
        The zonal wavenumber (order)
    eig_vecval: ndarray
        Eigenvalues & eigenvectors

    Returns
    -----
    mke : ndarray
        The mean kinetic energy
    mme : ndarray
        The mean magnetic energy

    Notes
    -----
    This function is based on eq. (21) in Nakashima & Yoshida
    (submitted)[1]_.

    """

    size_mat: int = eig_vecval.shape[1]
    size_submat: int = int(size_mat/2)

    mke: np.ndarray = np.zeros(size_mat)
    mme: np.ndarray = np.zeros(size_mat)

    n_degree: int
    nn1: int
    for i_n in range(size_submat):
        n_degree = m_order + i_n
        nn1 = n_degree * (n_degree+1)

        mke += nn1 * (np.abs(eig_vecval[i_n, :])**2)
        mme += nn1 * (np.abs(eig_vecval[size_submat+i_n, :])**2)
    #

    return mke, mme
#


def calc_qty(m_order: int,
             e_eta: float,
             eig_vecval: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate various physical quantities from eigenvectors

    Parameters
    -----
    m_order : int
        The zonal wavenumber (order)
    e_eta : float
        The magnetic Ekman number
    eig_vecval: np.ndarray
        Eigenvalues & normalized eigenvectors

    Returns
    -----
    mke : ndarray
        The mean kinetic energy
    mme : ndarray
        The mean magnetic energy
    ohm : ndarray
        Ohmic dissipation
    sym : ndarray
        The symmetry of eigenmodes

    Notes
    -----
    This function is based on eq. () in

    """

    size_mat: int = eig_vecval.shape[1]
    size_submat: int = int(size_mat/2)

    mke: np.ndarray
    mme: np.ndarray
    mke, mme = calc_ene(m_order, eig_vecval)

    ohm: np.ndarray = np.zeros(size_mat)
    if e_eta != 0:
        n_degree: int
        nn1: int
        for i_n in range(size_submat):
            n_degree = m_order + i_n
            nn1 = n_degree * (n_degree+1)

            ohm += (nn1**2) * (
                np.abs(eig_vecval[size_submat+i_n, :])**2)
        #
        ohm *= e_eta
    #

    even: np.ndarray = np.zeros(size_mat)
    odd: np.ndarray = np.zeros(size_mat)
    sym: np.ndarray = np.full(size_mat, str(), dtype=object)
    for i_n in range(int(size_submat/2)):
        even += np.abs(eig_vecval[2*i_n, :])
        odd += np.abs(eig_vecval[2*i_n+1, :])
    #
    for i_mode in range(size_mat):
        if even[i_mode] > odd[i_mode]:
            sym[i_mode] = 'sinuous'
        else:
            sym[i_mode] = 'varicose'
        #
    #

    return mke, mme, ohm, sym
#


def check_eig(m_order: int,
              criterion_c: tuple[int, float],
              alpha: float,
              eig_vecval: np.ndarray) -> np.ndarray:
    """Checks the validity of eigenmodes

    Parameters
    -----
    m_order : int
        The zonal wavenumber (order)
    criterion_c : tuple of int and float
        A criterion for convergence (degree, ratio)
    alpha : float
        The Lehnert number
    eig_vecval : ndarray
        Eigenvalues & normalized eigenvectors

    Returns
    -----
    check : ndarray
        validity

    Notes
    -----
    This function is based on eq. (20) in Nakashima & Yoshida
    (submitted)[1]_.

    """

    n_c: int
    r_c: float
    n_c, r_c = criterion_c

    size_mat: int = eig_vecval.shape[1]
    size_submat: int = int(size_mat/2)

    n_degree: int

    low_psi: np.ndarray = np.zeros(size_mat)
    low_a: np.ndarray = np.zeros(size_mat)
    high_psi: np.ndarray = np.zeros(size_mat)
    high_a: np.ndarray = np.zeros(size_mat)
    for i_n in range(size_submat):
        n_degree = m_order + i_n

        if n_degree <= n_c:
            low_psi += (np.abs(eig_vecval[i_n, :])**2)
            low_a += (np.abs(eig_vecval[size_submat+i_n, :])**2)
        else:
            high_psi += (np.abs(eig_vecval[i_n, :])**2)
            high_a += (np.abs(eig_vecval[size_submat+i_n, :])**2)
        #
    #

    check: np.ndarray
    if alpha != 0:
        check = (low_psi > high_psi*r_c) * (low_a > high_a*r_c)
    else:
        check = low_psi > high_psi*r_c
    #

    return check
#


def screening_eig(eig_vecval: np.ndarray,
                  phys_qtys: tuple[
                      np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                  check: np.ndarray) \
        -> tuple[np.ndarray,
                 tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Excludes invalid eigenmodes

    Parameters
    -----
    eig_vecval : ndarray
        Eigenvalues & normalized eigenvectors
    phys_qtys : tuple of ndarray
        mke, mme, ohm, sym
    check : ndarray
        Validity

    Returns
    -----
    eig_vecval : ndarray
        Eigenvalues & normalized eigenvectors
    phys_qtys : tuple of ndarray
        mke, mme, ohm, sym

    """

    mke: np.ndarray
    mme: np.ndarray
    ohm: np.ndarray
    sym: np.ndarray
    mke, mme, ohm, sym = phys_qtys

    size_mat: int = eig_vecval.shape[1]

    for i_mode in range(size_mat):
        if not check[i_mode]:
            eig_vecval[:, i_mode] = math.nan
            mke[i_mode] = math.nan
            mme[i_mode] = math.nan
            ohm[i_mode] = math.nan
            sym[i_mode] = None
        #
    #

    phys_qtys = (mke, mme, ohm, sym)

    return eig_vecval, phys_qtys
#
