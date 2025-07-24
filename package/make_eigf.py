"""Makes a eigenfunction from an eigenvector"""

import logging
import math

import numpy as np

from package.input_arg import input_int_within

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def choose_eigf(bundle: tuple[np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray],
                size_mat: int) \
        -> tuple[np.ndarray, np.ndarray, complex, int]:
    """Chooses eigenmodes which you want to plot

    Parameters
    -----
    bundle : tuple of ndarray
        A tuple of results
    size_mat : int
        The size of matrices

    Returns
    -----
    subbundle : tuple of ndarray and complex and int
        A tuple of results that you want to plot

    Raises
    -----
    Invalid eigenmode
        When you choose an eigenmode that can not be displayed.

    """

    psi_vec: np.ndarray
    vpa_vec: np.ndarray
    eig: np.ndarray
    mke: np.ndarray
    sym: np.ndarray
    psi_vec, vpa_vec, eig, mke, _, _, sym = bundle

    mode_list: list = []
    q_value: float
    for i_mode in range(size_mat):

        if math.isnan(eig[i_mode].real):
            continue
        #
        mode_list.append(i_mode)

        q_value = math.inf
        if eig[i_mode].imag != 0:
            q_value = math.fabs(
                eig[i_mode].real) / (-2*eig[i_mode].imag)
        #

        print(f'({i_mode+1:04})  '
              + f'[{eig[i_mode].real:8.5f},{eig[i_mode].imag:8.5f}] '
              + f'{sym[i_mode]:>9s}  Q={q_value:4.2f}  '
              + f'MKE={mke[i_mode]:4.2f}')
    #

    print('==============================')
    i_mode_min: int = min(mode_list) + 1
    i_mode_max: int = max(mode_list) + 1
    print(
        f'Please enter an integer (from {i_mode_min} to {i_mode_max})')

    chosen_int: int
    i_chosen: int
    while True:
        chosen_int = input_int_within(i_mode_min, i_mode_max)
        i_chosen = chosen_int - 1

        if i_chosen in mode_list:
            break
        #
        logger.error('Invalid eigenmode')
    #

    print(f'You chose: ({i_chosen+1:04})  '
          + f'[{eig[i_chosen].real:8.5f},{eig[i_chosen].imag:8.5f}] '
          + f'{sym[i_chosen]:>9s}  Q={q_value:4.2f}  '
          + f'MKE={mke[i_chosen]:4.2f}')
    print('==============================')

    subbundle: tuple[np.ndarray, np.ndarray, complex, int] \
        = (psi_vec[:, i_chosen], vpa_vec[:, i_chosen],
           eig[i_chosen], i_chosen)

    return subbundle
#


def make_eigf(psi_vec: np.ndarray,
              vpa_vec: np.ndarray,
              m_order: int,
              legendre_norm: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    """Makes an eigenfunction from an eigenvector

    Parameters
    -----
    psi_vec : ndarray
        An eigenvector of the stream function (psi)
    vpa_vec : ndarray
        An eigenvector of the vector potential (a)
    m_order : int
        The zonal wavenumber (order)
    legendre_norm : ndarray
        Values of associated Legendre polynomials at grid points

    Returns
    -----
    psi : ndarray
        An eigenfunction of the stream function (psi)
    vpa : ndarray
        An eigenfunction of the vector potential (a)

    """

    num_theta: int = legendre_norm.shape[1]
    size_submat: int = psi_vec.shape[0]

    psi: np.ndarray = np.zeros(num_theta, dtype=np.complex128)
    vpa: np.ndarray = np.zeros(num_theta, dtype=np.complex128)

    n_degree: int
    for i_n in range(size_submat):
        n_degree = m_order + i_n

        psi += psi_vec[i_n] * legendre_norm[n_degree, :]
        vpa += vpa_vec[i_n] * legendre_norm[n_degree, :]
    #

    sign: int = adjust_sign(psi, num_theta)

    psi *= sign
    vpa *= sign

    return psi, vpa
#


def make_eigf_grid(psi_vec: np.ndarray,
                   vpa_vec: np.ndarray,
                   m_order: int,
                   num_phi: int,
                   legendre_norm: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    """Makes a meshgrid of an eigenfunction from an eigenvector

    Parameters
    -----
    psi_vec : ndarray
        An eigenvector of the stream function (psi)
    vpa_vec : ndarray
        An eigenvector of the vector potential (a)
    m_order : int
        The zonal wavenumber (order)
    num_phi : int
        The number of the grid in the phi direction
    legendre_norm : ndarray
        Values of associated Legendre polynomials at grid points

    Returns
    -----
    psi_grid.real : ndarray
        A meshgrid of the stream function (psi)
    vpa_grid.real : ndarray
        A meshgrid of the vector potential (a)

    """

    num_theta: int = legendre_norm.shape[1]

    lin_theta: np.ndarray = np.linspace(0, math.pi, num_theta)
    lin_phi: np.ndarray = np.linspace(0, 2 * math.pi, num_phi)

    grid_phi: np.ndarray
    grid_theta: np.ndarray
    grid_phi, grid_theta \
        = np.meshgrid(lin_phi, lin_theta[1:-1])

    psi_grid: np.ndarray \
        = np.zeros_like(grid_theta, dtype=np.complex128)
    vpa_grid: np.ndarray \
        = np.zeros_like(grid_theta, dtype=np.complex128)

    psi: np.ndarray
    vpa: np.ndarray
    psi, vpa = make_eigf(psi_vec, vpa_vec, m_order, legendre_norm)

    psi_grid = np.meshgrid(lin_phi, psi[1:-1])[1]
    vpa_grid = np.meshgrid(lin_phi, vpa[1:-1])[1]

    phase: np.ndarray \
        = np.cos(m_order * grid_phi) + 1j*np.sin(m_order * grid_phi)

    psi_grid *= phase
    vpa_grid *= phase

    return psi_grid.real, vpa_grid.real
#


def amp_range(psi: np.ndarray,
              vpa: np.ndarray) -> tuple[float, float]:
    """Determines the range of amplitude in a 1D plot

    Parameters
    -----
    psi : ndarray
        An eigenfunction of the stream function (psi)
    vpa : ndarray
        An eigenfunction of the vector potential (a)

    Returns
    -----
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
#


def adjust_sign(psi: np.ndarray,
                num_theta: int) -> int:
    """Adjusts the sign of eigenfunctions

    Parameters
    -----
    psi : ndarray
        An eigenfunction of the stream function (psi)
    num_theta : int
        The number of the grid in the theta direction

    Returns
    -----
    sign : int
        The sign of the eigenfunction

    """

    width: int = int(num_theta*0.01)

    i_equator: int
    if num_theta % 2 == 1:
        i_equator = int((num_theta-1)/2)
    else:
        i_equator = int(num_theta/2)
    #
    equator: float = np.sum(psi.real[i_equator-width:i_equator])

    sign: int = np.sign(equator)

    return sign
#
