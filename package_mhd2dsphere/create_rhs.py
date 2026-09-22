"""A Python module to create the right-hand sides of the governing equations
for the (quasi-)linear simulation of two-dimensional (2D) incompressible
magnetohydrodynamic (MHD) waves on a rotating sphere under a toroidal
background field, B_phi = B_0 B(theta) sin(theta), and a background zonal flow,
U_phi = U_0 U(theta) sin(theta).

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
from package_common.common_types import ArrayComplex, ArrayFloat, cast
from package_common.default_logger import DefaultLogger
from package_common.utils_collocation import (create_cheb_diff_mat,
                                              spherical_laplacian_heinrichs)
from package_common.utils_debug import under_construction_log
from package_common.utils_name import create_function_name_logger
from package_common.utils_simulation import Rhs
from package_mhd2dsphere.typed_dict import DictRhsCommonParts


def wrapper_rhs(name: str,
                m_order: int = 0,
                *,
                list_m_order: list[int],
                linsp_mu: ArrayFloat,
                alpha: float,
                e_eta: float,
                rossby: float,
                switch_quasi_lin: bool) -> Rhs:
    """Wrapper function to create the right-hand side of the governing
    equations.

    Parameters
    ----------
    name : str
        The name of the field.
    m_order : int | None, optional, default 0.
        The given zonal wavenumber (order).
    list_m_order : list[int]
        The list of all the zonal wavenumbers (orders).
    linsp_mu : ArrayFloat
        The values of mu at grid points.
    alpha : float
        The Lehnert number.
    e_eta : float
        The magnetic Ekman number.
    rossby : float
        The Rossby number.
    switch_quasi_lin : bool
        The boolean value to switch whether to perform quasi-linear simulations
        or not.

    Returns
    -------
    Rhs
        The right-hand side of a governing equation.

    Warnings
    --------
    Invalid argument
            If m_order is 0 when `name` is 'psi' or 'mvp'.
    Unknown name
        If the name is undefined
    """

    if ((name == 'psi') or (name == 'mvp')) and (m_order == 0):
        logger: DefaultLogger = create_function_name_logger()
        logger.error('Invalid argument')

    size_submat: int = linsp_mu.shape[0]
    diff_mat: ArrayFloat = create_cheb_diff_mat(size_submat+2)

    hein: ArrayFloat = np.empty((size_submat, size_submat))
    laplacian: ArrayFloat = np.empty((size_submat, size_submat))
    for i_mu, mu in enumerate(linsp_mu):
        hein[:size_submat, i_mu] = np.array(
            [cast(float, heinrichs(i_n, mu)) for i_n in range(size_submat)]
        )
        laplacian[:size_submat, i_mu] = np.array(
            [cast(float, spherical_laplacian_heinrichs(m_order, i_n, mu))
             for i_n in range(size_submat)]
        )

    common_parts: DictRhsCommonParts = {
        "diff_mat": diff_mat,
        "heinrichs": hein,
        "laplacian": laplacian
    }

    if name == 'psi':
        def func_rhs_psi(fields_value: list[ArrayComplex | ArrayFloat],
                         time: float) -> ArrayComplex | ArrayFloat:
            return rhs_psi(fields_value,
                           list_m_order=list_m_order,
                           linsp_mu=linsp_mu,
                           m_order=m_order,
                           alpha=alpha,
                           rossby=rossby,
                           common_parts=common_parts)

        return func_rhs_psi

    elif name == 'mvp':
        def func_rhs_mvp(fields_value: list[ArrayComplex | ArrayFloat],
                         time: float) -> ArrayComplex | ArrayFloat:
            return rhs_mvp(fields_value,
                           list_m_order=list_m_order,
                           linsp_mu=linsp_mu,
                           m_order=m_order,
                           alpha=alpha,
                           e_eta=e_eta,
                           rossby=rossby,
                           common_parts=common_parts)

        return func_rhs_mvp

    elif name == 'bf_u_sin':
        def func_rhs_bf_u_sin(fields_value: list[ArrayComplex | ArrayFloat],
                              time: float) -> ArrayFloat:
            return rhs_bf_u_sin(fields_value,
                                list_m_order=list_m_order,
                                linsp_mu=linsp_mu,
                                switch_quasi_lin=switch_quasi_lin,
                                common_parts=common_parts)

        return func_rhs_bf_u_sin

    elif name == 'bf_b_sin':
        def func_rhs_bf_b_sin(fields_value: list[ArrayComplex | ArrayFloat],
                              time: float) -> ArrayFloat:
            return rhs_bf_b_sin(fields_value,
                                list_m_order=list_m_order,
                                linsp_mu=linsp_mu,
                                switch_quasi_lin=switch_quasi_lin,
                                common_parts=common_parts)

        return func_rhs_bf_b_sin

    logger: DefaultLogger = create_function_name_logger()
    logger.error('Unknown name')


def rhs_psi(fields_value: list[ArrayComplex | ArrayFloat],
            *,
            list_m_order: list[int],
            linsp_mu: ArrayFloat,
            m_order: int,
            alpha: float,
            rossby: float,
            common_parts: DictRhsCommonParts) -> ArrayComplex:
    """Create the right-hand side of the governing equation for the stream
    function for a given zonal wavenumber.

    Parameters
    ----------
    fields_value : list[ArrayComplex | ArrayFloat]
        The list of fields (list of psi, list of mvp, bf_u_sin, bf_b_sin).
    list_m_order : list[int]
        The list of all the zonal wavenumbers (orders).
    linsp_mu : ArrayFloat
        The values of mu at grid points.
    m_order : int
        The given zonal wavenumber (order).
    alpha : float
        The Lehnert number.
    rossby : float
        The Rossby number.
    common_parts : DictRhsCommonParts
        The common parts in the right-hand sides of the governing equations.

    Returns
    -------
    ArrayComplex
        The right-hand side of the governing equation for the stream function.
    """

    num_m: int = len(list_m_order)
    size_submat: int = linsp_mu.shape[0]

    list_psi: list[ArrayFloat | ArrayComplex] = fields_value[0:num_m]
    list_mvp: list[ArrayFloat | ArrayComplex] = fields_value[num_m:2*num_m]
    bf_u_sin: ArrayFloat = cast(ArrayFloat, fields_value[2*num_m])
    bf_b_sin: ArrayFloat = cast(ArrayFloat, fields_value[2*num_m+1])

    psi: ArrayFloat | ArrayComplex = list_psi[list_m_order.index(m_order)]
    mvp: ArrayFloat | ArrayComplex = list_mvp[list_m_order.index(m_order)]

    bf_u: ArrayFloat = bf_u_sin / np.sqrt(1-(linsp_mu**2))
    bf_b: ArrayFloat = bf_b_sin / np.sqrt(1-(linsp_mu**2))

    diff_mat: ArrayFloat = common_parts["diff_mat"]
    bf_u_sin_d: ArrayFloat = diff_mat @ np.concatenate(([0], bf_u_sin, [0]))
    bf_u_sin_d2: ArrayFloat = diff_mat @ bf_u_sin_d
    bf_u_shear: ArrayFloat = (
        bf_u_sin_d2[1:-1] * np.sqrt(1-(linsp_mu**2))
        - (2*linsp_mu/np.sqrt(1-(linsp_mu**2))) * bf_u_sin_d[1:-1]
        - bf_u / (1-(linsp_mu**2))
    )
    bf_b_sin_d: ArrayFloat = diff_mat @ np.concatenate(([0], bf_b_sin, [0]))
    bf_b_sin_d2: ArrayFloat = diff_mat @ bf_b_sin_d
    bf_b_shear: ArrayFloat = (
        bf_b_sin_d2[1:-1] * np.sqrt(1-(linsp_mu**2))
        - (2*linsp_mu/np.sqrt(1-(linsp_mu**2))) * bf_b_sin_d[1:-1]
        - bf_b / (1-(linsp_mu**2))
    )

    submat_11: ArrayFloat = np.zeros(
        (size_submat, size_submat), dtype=np.float64)
    submat_12: ArrayFloat = np.zeros(
        (size_submat, size_submat), dtype=np.float64)
    submat_b_11: ArrayFloat = np.zeros(
        (size_submat, size_submat), dtype=np.float64)

    h_n: float
    laplacian: float
    for i_l in range(size_submat):
        for i_n in range(size_submat):
            h_n = common_parts["heinrichs"][i_n, i_l]
            laplacian = common_parts["laplacian"][i_n, i_l]

            submat_11[i_l, i_n] = (
                rossby * bf_u[i_l] * laplacian + h_n
                - rossby * bf_u_shear[i_l] * h_n
            )
            submat_12[i_l, i_n] \
                = bf_b[i_l] * laplacian - bf_b_shear[i_l] * h_n
            submat_b_11[i_l, i_n] = laplacian

    submat_11 = np.linalg.solve(submat_b_11, submat_11).astype(np.float64)
    submat_12 = np.linalg.solve(submat_b_11, submat_12).astype(np.float64)

    submat_11 *= m_order
    submat_12 *= -m_order * alpha

    return cast(ArrayComplex, -1j * (submat_11 @ psi + submat_12 @ mvp))


def rhs_mvp(fields_value: list[ArrayComplex | ArrayFloat],
            *,
            list_m_order: list[int],
            linsp_mu: ArrayFloat,
            m_order: int,
            alpha: float,
            e_eta: float,
            rossby: float,
            common_parts: DictRhsCommonParts) -> ArrayComplex:
    """Create the right-hand side of the governing equation for the vector
    potential for a given zonal wavenumber.

    Parameters
    ----------
    fields_value : list[ArrayComplex | ArrayFloat]
        The list of fields (list of psi, list of mvp, bf_u_sin, and bf_b_sin).
    list_m_order : list[int]
        The list of zonal wavenumbers (orders).
    linsp_mu : ArrayFloat
        The values of mu at grid points.
    m_order : int
        The given zonal wavenumber (order).
    alpha : float
        The Lehnert number.
    e_eta : float
        The magnetic Ekman number.
    rossby : float
        The Rossby number.
    common_parts : DictRhsCommonParts
        The common parts in the right-hand sides of the governing equations.

    Returns
    -------
    ArrayComplex
        The right-hand side of the governing equation for the vector potential.
    """

    num_m: int = len(list_m_order)
    size_submat: int = linsp_mu.shape[0]

    list_psi: list[ArrayFloat | ArrayComplex] = fields_value[0:num_m]
    list_mvp: list[ArrayFloat | ArrayComplex] = fields_value[num_m:2*num_m]
    bf_u_sin: ArrayFloat = cast(ArrayFloat, fields_value[2*num_m])
    bf_b_sin: ArrayFloat = cast(ArrayFloat, fields_value[2*num_m+1])

    psi: ArrayFloat | ArrayComplex = list_psi[list_m_order.index(m_order)]
    mvp: ArrayFloat | ArrayComplex = list_mvp[list_m_order.index(m_order)]

    bf_u: ArrayFloat = bf_u_sin / np.sqrt(1-(linsp_mu**2))
    bf_b: ArrayFloat = bf_b_sin / np.sqrt(1-(linsp_mu**2))

    submat_21: ArrayFloat = np.zeros(
        (size_submat, size_submat), dtype=np.float64)
    submat_b_22: ArrayFloat = np.zeros(
        (size_submat, size_submat), dtype=np.float64)

    submat_22: ArrayComplex | ArrayFloat
    if e_eta != 0:
        submat_22 = np.zeros((size_submat, size_submat), dtype=np.complex128)
    else:
        submat_22 = np.zeros((size_submat, size_submat), dtype=np.float64)

    h_n: float
    laplacian: float
    for i_l in range(size_submat):
        for i_n in range(size_submat):
            h_n = common_parts["heinrichs"][i_n, i_l]
            laplacian = common_parts["laplacian"][i_n, i_l]

            submat_21[i_l, i_n] = bf_b[i_l] * h_n
            submat_22[i_l, i_n] = m_order * rossby * bf_u[i_l] * h_n
            if e_eta != 0:
                submat_22[i_l, i_n] += 1j * e_eta * laplacian
            submat_b_22[i_l, i_n] = h_n

    submat_21 = np.linalg.solve(submat_b_22, submat_21).astype(np.float64)
    if e_eta != 0:
        submat_22 = np.linalg.solve(
            submat_b_22, submat_22).astype(np.complex128)
    else:
        submat_22 = np.linalg.solve(submat_b_22, submat_22).astype(np.float64)

    submat_21 *= -m_order * alpha

    return cast(ArrayComplex, -1j * (submat_21 @ psi + submat_22 @ mvp))


def rhs_bf_u_sin(fields_value: list[ArrayComplex | ArrayFloat],
                 *,
                 list_m_order: list[int],
                 linsp_mu: ArrayFloat,
                 switch_quasi_lin: bool,
                 common_parts: DictRhsCommonParts) -> ArrayFloat:
    """Create the right-hand side of the governing equation for the background
    zonal flow.

    Parameters
    ----------
    fields_value : list[ArrayComplex | ArrayFloat]
        The list of fields (list of psi, list of mvp, bf_u_sin, and bf_b_sin).
    list_m_order : list[int]
        The list of all the zonal wavenumbers (orders).
    linsp_mu : ArrayFloat
        The values of mu at grid points.
    switch_quasi_lin: bool
        The boolean value to switch whether to perform quasi-linear simulations
        or not.
    common_parts : DictRhsCommonParts
        The common parts in the right-hand sides of the governing equations.

    Returns
    -------
    rhs : ArrayFloat
        The right-hand side of the governing equation for the background zonal
        flow.
    """

    num_m: int = len(list_m_order)

    list_psi: list[ArrayFloat | ArrayComplex] = fields_value[0:num_m]
    list_mvp: list[ArrayFloat | ArrayComplex] = fields_value[num_m:2*num_m]
    bf_u_sin: ArrayFloat = cast(ArrayFloat, fields_value[2*num_m])
    bf_b_sin: ArrayFloat = cast(ArrayFloat, fields_value[2*num_m+1])

    rhs: ArrayFloat = np.zeros_like(bf_u_sin)
    if not switch_quasi_lin:
        return rhs

    psi: ArrayFloat | ArrayComplex
    mvp: ArrayFloat | ArrayComplex
    for i_m, m_order in enumerate(list_m_order):
        psi = list_psi[i_m]
        mvp = list_mvp[i_m]

        under_construction_log()
        rhs += 0

    return rhs


def rhs_bf_b_sin(fields_value: list[ArrayComplex | ArrayFloat],
                 *,
                 list_m_order: list[int],
                 linsp_mu: ArrayFloat,
                 switch_quasi_lin: bool,
                 common_parts: DictRhsCommonParts) -> ArrayFloat:
    """Create the right-hand side of the governing equation for the toroidal
    background field.

    Parameters
    ----------
    fields_value : list[ArrayComplex | ArrayFloat]
        The list of fields (list of psi, list of mvp, bf_u_sin, and bf_b_sin).
    list_m_order : list[int]
        The list of all the zonal wavenumbers (orders).
    linsp_mu : ArrayFloat
        The values of mu at grid points.
    switch_quasi_lin : bool
        The boolean value to switch whether to perform quasi-linear simulations
        or not.
    common_parts : DictRhsCommonParts
        The common parts in the right-hand sides of the governing equations.

    Returns
    -------
    rhs : ArrayFloat
        The right-hand side of the governing equation for the background
        magnetic field.
    """

    num_m: int = len(list_m_order)

    list_psi: list[ArrayFloat | ArrayComplex] = fields_value[0:num_m]
    list_mvp: list[ArrayFloat | ArrayComplex] = fields_value[num_m:2*num_m]
    bf_u_sin: ArrayFloat = cast(ArrayFloat, fields_value[2*num_m])
    bf_b_sin: ArrayFloat = cast(ArrayFloat, fields_value[2*num_m+1])

    rhs: ArrayFloat = np.zeros_like(bf_b_sin)
    if not switch_quasi_lin:
        return rhs

    psi: ArrayFloat | ArrayComplex
    mvp: ArrayFloat | ArrayComplex
    for i_m, m_order in enumerate(list_m_order):
        psi = list_psi[i_m]
        mvp = list_mvp[i_m]

        under_construction_log()
        rhs += 0

    return rhs
