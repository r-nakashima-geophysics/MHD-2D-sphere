"""A Python module to make an eigenfunction from an eigenvector of the
eigenvalue problem of two-dimensional (2D) incompressible magnetohydrodynamic
(MHD) waves on a rotating sphere under a toroidal background field, B_phi = B_0
B(theta) sin(theta), and a background zonal flow, U_phi = U_0 U(theta)
sin(theta)."""

import numpy as np

from package_common.calc_heinrichs import heinrichs
from package_common.common_types import ArrayComplex, ArrayFloat, ArrayStr
from package_common.default_logger import DefaultLogger
from package_common.spectral_deform import ComplexCoordinate
from package_common.utils_input import input_value_within
from package_common.utils_name import create_function_name_logger
from package_mhd2dsphere._make_legendre import load_legendre
from package_mhd2dsphere.typed_dict import (DictBackgroundField,
                                            DictEigenmodeInfo, DictResult)


def create_basis(
        m_order: int,
        n_t: int,
        lin_theta: ArrayFloat,
        *,
        background_field: DictBackgroundField) -> ArrayFloat | ArrayComplex:
    """Create basis functions for eigenfunctions

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    n_t : int
        The truncation degree.
    lin_theta : ArrayFloat
        The values of theta at grid points.
    background_field : DictBackgroundField
        The background field.

    Returns
    -------
    basis : ArrayFloat | ArrayComplex
        The values of basis functions at grid points.
    """

    num_theta: int = lin_theta.shape[0]

    basis: ArrayFloat | ArrayComplex
    if background_field['NY24']:
        basis = load_legendre(m_order, n_t, num_theta)
        return basis

    size_submat: int = n_t + 1
    mu_complex: ComplexCoordinate = background_field['MU']

    basis = np.empty((size_submat, num_theta), dtype=np.complex128)

    x_pos: float
    s_pos: complex | float
    guess: complex
    for i_theta, theta in enumerate(lin_theta):
        x_pos = np.cos(theta)
        if i_theta == 0:
            guess = x_pos + 1j * 0
        else:
            guess = s_pos
        s_pos = mu_complex.inverse(x_pos, guess=guess)

        basis[:, i_theta] = np.array(
            [heinrichs(i_n, s_pos) for i_n in range(size_submat)]
        )

    return basis


def choose_eigfunc(results: DictResult,
                   size_mat: int) -> DictEigenmodeInfo:
    """Choose eigenmodes which you want to plot

    Parameters
    ----------
    results : DictResult
        A dictionary of results of the eigenvalue problem.
    size_mat : int
        The size of matrices.

    Returns
    -------
    result : DictEigenmodeInfo
        A dictionary of result of an eigenmode which you chose.

    Warnings
    --------
    Invalid eigenmode
        When you choose an eigenmode that can not be plotted.
    """

    logger: DefaultLogger = create_function_name_logger()

    eig: ArrayComplex = results['eig']
    pke: ArrayFloat = results['phys_qtys']['pke']
    pme: ArrayFloat = results['phys_qtys']['pme']
    sym: ArrayStr = results['phys_qtys']['sym']

    mode_list: list = []
    q_value: float
    for i_mode in range(size_mat):

        if np.isnan(eig[i_mode].real):
            continue

        mode_list.append(i_mode)

        q_value = np.inf
        if eig[i_mode].imag != 0:
            q_value = np.abs(eig[i_mode].real) / (-2*eig[i_mode].imag)

        print(f'({i_mode+1:04})  '
              + f'[{eig[i_mode].real:8.5f},{eig[i_mode].imag:8.5f}] '
              + f'{sym[i_mode]:>9s}  Q={q_value:4.2f}  '
              + f'PKE={pke[i_mode]:4.2f}  PME={pme[i_mode]:4.2f}')

    print('==============================')
    i_mode_min: int = min(mode_list) + 1
    i_mode_max: int = max(mode_list) + 1
    print(f'Please enter an integer (between {i_mode_min} and {i_mode_max})')

    chosen_int: int
    i_chosen: int
    while True:
        chosen_int = input_value_within(i_mode_min, i_mode_max, int)
        i_chosen = chosen_int - 1

        if i_chosen in mode_list:
            break

        logger.warning('Invalid eigenmode')

    q_value = np.inf
    if eig[i_chosen].imag != 0:
        q_value = np.abs(eig[i_chosen].real) / (-2*eig[i_chosen].imag)
    print(f'You chose: ({i_chosen+1:04})  '
          + f'[{eig[i_chosen].real:8.5f},{eig[i_chosen].imag:8.5f}] '
          + f'{sym[i_chosen]:>9s}  Q={q_value:4.2f}  '
          + f'PKE={pke[i_chosen]:4.2f}  PME={pme[i_chosen]:4.2f}')
    print('==============================')

    result: DictEigenmodeInfo = {
        'i_mode': i_chosen,
        'eig': results['eig'][i_chosen],
        'vec_psi': results['vec_psi'][:, i_chosen],
        'vec_vpa': results['vec_vpa'][:, i_chosen],
        'pke': results['phys_qtys']['pke'][i_chosen],
        'pme': results['phys_qtys']['pme'][i_chosen],
        'psm': results['phys_qtys']['psm'][i_chosen],
        'pse': results['phys_qtys']['pse'][i_chosen],
        'ohm': results['phys_qtys']['ohm'][i_chosen],
        'sym': results['phys_qtys']['sym'][i_chosen]
    }

    return result


def make_eigfunc(result: DictEigenmodeInfo,
                 m_order: int,
                 lin_theta: ArrayFloat,
                 basis_func: ArrayFloat | ArrayComplex,
                 *,
                 background_field: DictBackgroundField) \
        -> tuple[ArrayComplex, ArrayComplex]:
    """Make an eigenfunction from an eigenvector

    Parameters
    ----------
    result : DictEigenmodeInfo
        A dictionary of result of an eigenmode which you chose.
    m_order : int
        The zonal wavenumber (order).
    lin_theta : ArrayFloat
        The values of theta at grid points.
    basis_func : ArrayFloat | ArrayComplex
        The values of basis functions at grid points.
    background_field : DictBackgroundField
        The background field.

    Returns
    ----------
    psi : ArrayComplex
        The stream function (psi).
    vpa : ArrayComplex
        The vector potential (a).
    """

    vec_psi: ArrayComplex = result['vec_psi']
    vec_vpa: ArrayComplex = result['vec_vpa']

    size_submat: int = vec_psi.shape[0]
    num_theta: int = lin_theta.shape[0]

    psi: ArrayComplex = np.zeros_like(lin_theta, dtype=np.complex128)
    vpa: ArrayComplex = np.zeros_like(lin_theta, dtype=np.complex128)

    if background_field['NY24']:
        n_degree: int
        for i_n in range(size_submat):
            n_degree = m_order + i_n

            psi += vec_psi[i_n] * basis_func[n_degree, :]
            vpa += vec_vpa[i_n] * basis_func[n_degree, :]
    else:
        for i_theta in range(num_theta):

            psi[i_theta] = basis_func[:, i_theta] @ vec_psi
            vpa[i_theta] = basis_func[:, i_theta] @ vec_vpa

    sign: int = adjust_sign(psi, num_theta)

    psi *= sign
    vpa *= sign

    return psi, vpa


def make_eigfunc_grid(result: DictEigenmodeInfo,
                      m_order: int,
                      lin_theta: ArrayFloat,
                      lin_phi: ArrayFloat,
                      basis_func: ArrayFloat | ArrayComplex,
                      *,
                      background_field: DictBackgroundField) \
        -> tuple[ArrayFloat, ArrayFloat]:
    """Make a meshgrid of an eigenfunction from an eigenvector

    Parameters
    ----------
    result : DictEigenmodeInfo
        A dictionary of result of an eigenmode which you chose.
    m_order : int
        The zonal wavenumber (order).
    lin_theta : ArrayFloat
        The values of theta at grid points.
    lin_phi : ArrayFloat
        The values of phi at grid points.
    basis_func : ArrayFloat | ArrayComplex
        The values of basis functions at grid points.
    background_field : DictBackgroundField
        The background field.

    Returns
    ----------
    psi_grid.real : ArrayFloat
        The stream function (psi).
    vpa_grid.real : ArrayFloat
        The vector potential (a).
    """

    psi: ArrayComplex
    vpa: ArrayComplex
    psi, vpa = make_eigfunc(result, m_order, lin_theta, basis_func,
                            background_field=background_field)

    grid_phi: ArrayFloat
    grid_phi, _ = np.meshgrid(lin_phi, lin_theta[1:-1])

    psi_grid: ArrayComplex
    vpa_grid: ArrayComplex
    _, psi_grid = np.meshgrid(lin_phi, psi[1:-1])
    _, vpa_grid = np.meshgrid(lin_phi, vpa[1:-1])

    phase: ArrayComplex \
        = np.cos(m_order * grid_phi) + 1j*np.sin(m_order * grid_phi)

    psi_grid *= phase
    vpa_grid *= phase

    return psi_grid.real, vpa_grid.real


def adjust_sign(psi: ArrayComplex,
                num_theta: int) -> int:
    """Adjust the sign of eigenfunctions

    Parameters
    ----------
    psi : ArrayComplex
        The stream function (psi).
    num_theta : int
        The number of the grid in the theta direction.

    Returns
    -------
    sign : int
        The sign of the eigenfunction.

    Warnings
    --------
    The adjustment of the sign of the eigenfunction is failed
        When the sum of the stream function around the equator is zero.
    """

    width: int = max(int(num_theta*0.01), 1)

    i_equator: int
    if num_theta % 2 == 1:
        i_equator = int((num_theta-1)/2)
    else:
        i_equator = int(num_theta/2)

    equator: float = np.sum(psi.real[i_equator-width:i_equator])

    if equator > 0:
        return 1
    elif equator < 0:
        return -1
    else:
        logger: DefaultLogger = create_function_name_logger()
        logger.warning(
            'The adjustment of the sign of the eigenfunction is failed')

    return 1


def amp_range(psi: ArrayComplex,
              vpa: ArrayComplex) -> tuple[float, float]:
    """Determine the range of amplitude in a 1D plot.

    Parameters
    ----------
    psi : ArrayComplex
        The stream function (psi).
    vpa : ArrayComplex
        The vector potential (a).

    Returns
    ----------
    amp_max : float
        The maximum value of the amplitude of the eigenfunction
    amp_min : float
        The minimum value of the amplitude of the eigenfunction

    """

    factor: float = 1.5

    psi_real_max: float = np.nanmax(np.abs(psi.real))
    vpa_real_max: float = np.nanmax(np.abs(vpa.real))
    psi_imag_max: float = np.nanmax(np.abs(psi.imag))
    vpa_imag_max: float = np.nanmax(np.abs(vpa.imag))

    amp_max: float = max(psi_real_max, vpa_real_max,
                         psi_imag_max, vpa_imag_max)
    amp_min: float = -amp_max

    amp_max *= factor
    amp_min *= factor

    return amp_max, amp_min
