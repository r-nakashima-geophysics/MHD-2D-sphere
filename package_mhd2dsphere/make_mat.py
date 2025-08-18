"""A Python module to make the matrices for the eigenvalue problem of
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

from package_common.common_types import ArrayComplex, ArrayFloat


def make_submat(m_order: int,
                size_submat: int,
                /,
                switch_ny24: bool = False) -> tuple[ArrayFloat,
                                                    ArrayFloat,
                                                    ArrayFloat,
                                                    ArrayFloat]:
    """Make the submatrices.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    size_submat : int
        The size of submatrices.
    switch_ny24 : bool, optional, default False
        The boolean value to switch whether to follow Nakashima &
        Yoshida (2024)[1]_ or not.

    Returns
    ----------
    submat_11 : ArrayFloat
        The (1,1)th block matrix.
    submat_12 : ArrayFloat
        The (1,2)th block matrix.
    submat_21 : ArrayFloat
        The (2,1)th block matrix.
    submat_22 : ArrayFloat
        The (2,2)th block matrix.

    Notes
    ----------
    This function is based on eq. (22) in Nakashima & Yoshida
    (2024)[1]_.
    """

    if switch_ny24:
        n_t: int = size_submat + m_order - 1
        lin_n: ArrayFloat = np.linspace(m_order, n_t, size_submat)

        knm: ArrayFloat = np.sqrt(
            (lin_n-m_order) * (lin_n+m_order) / ((2*lin_n-1)*(2*lin_n+1)))

        submat_11: ArrayFloat \
            = np.zeros((size_submat, size_submat), dtype=np.float64)
        for i_submat in range(size_submat):
            submat_11[i_submat, i_submat] \
                = 1 / ((m_order+i_submat)*(m_order+1+i_submat))

        submat_12: ArrayFloat \
            = np.zeros((size_submat, size_submat), dtype=np.float64)
        for i_submat in range(size_submat-1):
            submat_12[i_submat, i_submat+1] \
                = (m_order-1+i_submat) * (m_order+4+i_submat) \
                * knm[1+i_submat] \
                / ((m_order+i_submat)*(m_order+1+i_submat))
            submat_12[i_submat+1, i_submat] \
                = (m_order-2+i_submat) * (m_order+3+i_submat) \
                * knm[1+i_submat] \
                / ((m_order+1+i_submat)*(m_order+2+i_submat))

        submat_21: ArrayFloat \
            = np.zeros((size_submat, size_submat), dtype=np.float64)
        for i_submat in range(size_submat-1):
            submat_21[i_submat, i_submat+1] = knm[1+i_submat]
            submat_21[i_submat+1, i_submat] = knm[1+i_submat]

        submat_22: ArrayFloat \
            = np.zeros((size_submat, size_submat), dtype=np.float64)
        for i_submat in range(size_submat):
            submat_22[i_submat, i_submat] \
                = (m_order+i_submat) * (m_order+1+i_submat)

        return submat_11, submat_12, submat_21, submat_22
    else:
        return None, None, None, None  # under construction


def make_mat(m_order: int,
             e_eta: float,
             submatrices: tuple[ArrayFloat,
                                ArrayFloat,
                                ArrayFloat,
                                ArrayFloat],
             alpha: float,
             /,
             switch_ny24: bool = False) -> ArrayFloat | ArrayComplex:
    """Make the total matrix.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    e_eta : float
        The magnetic Ekman number.
    submatrices : tuple[ArrayFloat, ArrayFloat, ArrayFloat, ArrayFloat]
        The (1,1)th, (1,2)th, (2,1)th, and (2,2)th block matrices.
    alpha : float
        The Lehnert number.
    switch_ny24 : bool, optional, default False
        The boolean value to switch whether to follow Nakashima &
        Yoshida (2024)[1]_ or not.

    Returns
    ----------
    mat: ArrayFloat | ArrayComplex
        The total matrix.

    Notes
    ----------
    This function is based on eq. (22) in Nakashima & Yoshida
    (2024)[1]_.
    """

    submat_11: ArrayFloat
    submat_12: ArrayFloat
    submat_21: ArrayFloat
    submat_22: ArrayFloat
    mat: ArrayFloat | ArrayComplex

    if switch_ny24:
        submat_11, submat_12, submat_21, submat_22 = submatrices

        size_submat: int = submat_11.shape[0]
        size_mat: int = 2 * size_submat

        if e_eta == 0:
            mat = np.zeros((size_mat, size_mat), dtype=np.float64)
        else:
            mat = np.zeros((size_mat, size_mat), dtype=np.complex128)

        mat[0*size_submat:1*size_submat, 0*size_submat:1*size_submat] \
            = -m_order * submat_11
        mat[0*size_submat:1*size_submat, 1*size_submat:2*size_submat] \
            = -m_order * alpha * submat_12

        mat[1*size_submat:2*size_submat, 0*size_submat:1*size_submat] \
            = -m_order * alpha * submat_21
        if e_eta != 0:
            mat[1*size_submat:2*size_submat, 1*size_submat:2*size_submat] \
                = -1j * e_eta * submat_22
    else:
        mat = None  # under construction

    return mat
