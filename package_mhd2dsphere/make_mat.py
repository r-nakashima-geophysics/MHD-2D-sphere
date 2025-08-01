"""Makes matrices for the eigenvalue problem of 2D MHD waves on a
rotating sphere under the non-Malkus field
B_phi = B_0 sin(theta) cos(theta)

References
----------
[1] Nakashima & Yoshida (submitted)

"""

import numpy as np


def make_submat(m_order: int,
                size_submat: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Makes submatrices

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order)
    size_submat : int
        The size of submatrices

    Returns
    ----------
    submat_r : ndarray
        The (1,1)th submatrix
    submat_k1 : ndarray
        The (1,2)th submatrix
    submat_k2 : ndarray
        The (2,1)th submatrix
    submat_d : ndarray
        The (2,2)th submatrix

    Notes
    ----------
    This function is based on eq. (22) in Nakashima & Yoshida
    (submitted)[1]_.

    """

    n_t: int = size_submat + m_order - 1
    lin_n: np.ndarray = np.linspace(m_order, n_t, size_submat)

    knm: np.ndarray = np.sqrt(
        (lin_n-m_order) * (lin_n+m_order) / ((2*lin_n-1)*(2*lin_n+1)))

    submat_r: np.ndarray = np.zeros((size_submat, size_submat))
    for i_submat in range(size_submat):
        submat_r[i_submat, i_submat] \
            = 1 / ((m_order+i_submat)*(m_order+1+i_submat))

    submat_k1: np.ndarray = np.zeros((size_submat, size_submat))
    for i_submat in range(size_submat-1):
        submat_k1[i_submat, i_submat+1] \
            = (m_order-1+i_submat) * (m_order+4+i_submat) \
            * knm[1+i_submat] \
            / ((m_order+i_submat)*(m_order+1+i_submat))
        submat_k1[i_submat+1, i_submat] \
            = (m_order-2+i_submat) * (m_order+3+i_submat) \
            * knm[1+i_submat] \
            / ((m_order+1+i_submat)*(m_order+2+i_submat))

    submat_k2: np.ndarray = np.zeros((size_submat, size_submat))
    for i_submat in range(size_submat-1):
        submat_k2[i_submat, i_submat+1] = knm[1+i_submat]
        submat_k2[i_submat+1, i_submat] = knm[1+i_submat]

    submat_d: np.ndarray = np.zeros((size_submat, size_submat))
    for i_submat in range(size_submat):
        submat_d[i_submat, i_submat] \
            = (m_order+i_submat) * (m_order+1+i_submat)

    return submat_r, submat_k1, submat_k2, submat_d
#


def make_mat(m_order: int,
             e_eta: float,
             submatrices: tuple[
                 np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             alpha: float) -> np.ndarray:
    """Makes the total matrix

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order)
    e_eta : float
        The magnetic Ekman number
    submatrices : tuple of ndarray
        submat_r, submat_k1, submat_k2, submat_d
    alpha : float
        The Lehnert number

    Returns
    ----------
    mat: np.ndarray
        The total matrix

    Notes
    ----------
    This function is based on eq. (22) in Nakashima & Yoshida
    (submitted)[1]_.

    """

    submat_r: np.ndarray
    submat_k1: np.ndarray
    submat_k2: np.ndarray
    submat_d: np.ndarray
    submat_r, submat_k1, submat_k2, submat_d = submatrices

    size_submat: int = submat_r.shape[0]
    size_mat: int = 2 * size_submat

    mat: np.ndarray
    if e_eta == 0:
        mat = np.zeros((size_mat, size_mat))
    else:
        mat = np.zeros((size_mat, size_mat), dtype=np.complex128)

    mat[0*size_submat:1*size_submat, 0*size_submat:1*size_submat] \
        = -m_order * submat_r
    mat[0*size_submat:1*size_submat, 1*size_submat:2*size_submat] \
        = -m_order * alpha * submat_k1

    mat[1*size_submat:2*size_submat, 0*size_submat:1*size_submat] \
        = -m_order * alpha * submat_k2
    if e_eta != 0:
        mat[1*size_submat:2*size_submat, 1*size_submat:2*size_submat] \
            = -1j * e_eta * submat_d

    return mat
#
