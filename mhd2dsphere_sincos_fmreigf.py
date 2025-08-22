"""A code for 2D ideal MHD waves on a rotating sphere under the
non-Malkus field B_phi = B_0 sin(theta) cos(theta)

Plots a figure of the approximate eigenfunction of fast magnetic
Rossby (MR) waves.

Raises
----------
ALPHA should be smaller than 0.5
    If ALPHA is larger than 0.5.

Notes
----------
Parameters other than command line arguments are described below.

References
----------
[1] Ryosuke Nakashima, Shigeo Yoshida, Two-dimensional ideal
magnetohydrodynamic waves on a rotating sphere under a non-Malkus field:
I. Continuous spectrum and its ray-theoretical interpretation.
Geophysical & Astrophysical Fluid Dynamics 118(5-6), 387-440 (2024).
doi: 10.1080/03091929.2024.2384388
"""

import logging
import math
import os
import sys
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from scipy.special import obl_ang1, pro_ang1

from package_common.decorator_yesno import exe_yes_continue
from package_mhd2dsphere.dispersion_fmr import dispersion_fmr
from package_mhd2dsphere.make_eigf import (adjust_sign, amp_range, choose_eigf,
                                           make_eigf)
from package_mhd2dsphere.make_legendre import load_legendre
from package_mhd2dsphere.solve_eig import wrapper_solve_eig

# ========== Parameters ==========

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = 0.1

# The truncation degree
N_T: Final[int] = 2000

# The number of the grid in the theta direction
NUM_THETA: Final[int] = 361

# A criterion for convergence
# degree
N_C: Final[int] = int(N_T/2)
# ratio
R_C: Final[float] = 100

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] = Path(
    '.') / 'fig' / 'MHD2Dsphere_sincos_fmreigf'
NAME_FIG: Final[str] = 'MHD2Dsphere_sincos_fmreigf' \
    + f'_m{M_ORDER}a{ALPHA}N{N_T}th{NUM_THETA}'
NAME_FIG_SUFFIX: Final[str] = '.png'
FIG_DPI: Final[int] = 600

# ================================

# The magnetic Ekman number
E_ETA: Final[float] = 0

CRITERION_C: Final[tuple[int, float]] = (N_C, R_C)

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

LIN_THETA: Final[np.ndarray] = np.linspace(0, math.pi, NUM_THETA)


@exe_yes_continue
def wrapper_choose_eigf(
        bundle: tuple[np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray]) -> None:
    """A wrapper of a function to choose eigenmodes which you want to
    plot

    Parameters
    ----------
    bundle : tuple of ndarray
        A tuple of results

    """

    psi_vec: np.ndarray
    vpa_vec: np.ndarray
    eig: complex
    i_chosen: int
    psi_vec, vpa_vec, eig, i_chosen = choose_eigf(bundle, SIZE_MAT)

    wrapper_plot_fmreigf(psi_vec, vpa_vec, eig, i_chosen)
#


def wrapper_plot_fmreigf(psi_vec: np.ndarray,
                         vpa_vec: np.ndarray,
                         eig: complex,
                         i_mode: int) -> None:
    """A wrapper of functions to plot a figure of the eigenfunction of a
    fast magnetic Rossby (MR) wave.

    Parameters
    ----------
    psi_vec : ndarray
        An eigenvector of the stream function (psi)
    vpa_vec : ndarray
        An eigenvector of the vector potential (a)
    eig : complex
        An eigenvalue
    i_mode : int
        The index of a mode that you chose

    """

    psi: np.ndarray
    vpa: np.ndarray
    psi, vpa = make_eigf(psi_vec, vpa_vec, M_ORDER, PNM_NORM)

    plot_ns_fmreigf(psi, vpa, eig, i_mode)

    plt.show()
#


def plot_ns_fmreigf(psi: np.ndarray,
                    vpa: np.ndarray,
                    eig: complex,
                    i_mode: int) -> None:
    """Plots a figure of the eigenfunction of a fast magnetic Rossby
    (MR) wave.

    Parameters
    ----------
    psi : ndarray
        An eigenfunction of the stream function (psi)
    vpa : ndarray
        An eigenfunction of the vector potential (a)
    eig : complex
        An eigenvalue
    i_mode : int
        The index of a mode that you chose

    """

    fig: plt.Figure
    axis: plt.Axes
    fig, axis = plt.subplots(figsize=(7, 4))

    axis.plot(LIN_THETA, psi.real, color='red',
              label=r'stream function $\tilde{\psi}$')
    axis.plot(LIN_THETA, vpa.real, color='blue',
              label=r'vector potential $\mathrm{sgn}(\alpha)\tilde{a}/'
              + r'\sqrt{\rho_0\mu_\mathrm{m}}$')
    if np.nanmax(np.abs(psi.imag)) > 0:
        axis.plot(LIN_THETA, psi.imag, color='red', linestyle=':')

    if np.nanmax(np.abs(vpa.imag)) > 0:
        axis.plot(LIN_THETA, vpa.imag, color='blue', linestyle=':')

    eig_fmr: float
    eigf_fmr: np.ndarray
    c2_spheroidal: float
    eig_fmr, eigf_fmr, c2_spheroidal = calc_fmreigf(i_mode)

    psi_max: float = np.nanmax(np.abs(psi.real))
    max_eigf_fmr: float = np.nanmax(np.abs(eigf_fmr))
    eigf_fmr = eigf_fmr * (psi_max/max_eigf_fmr)

    if c2_spheroidal >= 0:
        axis.plot(LIN_THETA, eigf_fmr, color='black',
                  linestyle=':', linewidth=2.5,
                  label=r'$\mathrm{S}_{mn}(c,\mu)/\sqrt{\Lambda}$')
    else:
        axis.plot(LIN_THETA, eigf_fmr, color='black',
                  linestyle=':', linewidth=2.5,
                  label=r'$\mathrm{S}_{mn}(-\mathrm{i}c,\mu)/\sqrt{\Lambda}$')

    max_amp, min_amp = amp_range(psi, vpa)

    axis.grid()
    axis.set_xlim(0, math.pi)
    axis.set_xticks([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi])
    axis.set_xticklabels(['$0$', '$45$', '$90$', '$135$', '$180$'])
    axis.set_ylim(min_amp, max_amp)

    axis.set_xlabel('colatitude [degree]', fontsize=16)
    axis.set_ylabel('amplitude', fontsize=16)
    axis.set_title(
        r'$\lambda=$' + f' {eig.real:8.5f}, '
        + r'$\lambda_\mathrm{approx}=$'
        + f' {eig_fmr.real:8.5f}, ' + r'$c^2=$'
        + f' {c2_spheroidal:8.5f}', fontsize=16)

    fig.suptitle(
        r'Eigenfunction [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
        + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}',
        fontsize=16)

    leg: plt.Legend = axis.legend(loc='best', fontsize=12)
    leg.get_frame().set_alpha(1)

    axis.tick_params(labelsize=14)
    axis.minorticks_on()

    fig.tight_layout()

    name_fig_full: str = NAME_FIG + f'_{i_mode+1}' + NAME_FIG_SUFFIX

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / name_fig_full
    fig.savefig(path_fig, dpi=FIG_DPI)
#


def calc_fmreigf(i_mode: int) -> tuple[float, np.ndarray, float]:
    """Makes the approximate eigenfunction of fast magnetic Rossby
    (MR) waves.

    Parameters
    ----------
    i_mode : int
        The index of a mode that you chose

    Returns
    ----------
    eig : float
        An approximate eigenvalue of fast magnetic Rossby (MR) waves
    fmreigf : ndarray
        An approximate eigenfunction of fast magnetic Rossby (MR) waves
    c2_spheroidal : float
        c^2 of the angular spheroidal wave function

    """

    n_degree: int = i_mode + M_ORDER

    eig: float = root(
        dispersion_fmr, -0.01,
        args=(M_ORDER, n_degree, ALPHA, 'sph'), method='hybr').x[0].real

    ma2: float = (M_ORDER**2) * (ALPHA**2)
    c2_spheroidal: float = (ma2/(eig**2)) * (7+M_ORDER/eig)

    critical: np.ndarray = (eig**2) - ma2*(np.cos(LIN_THETA)**2)

    sqrt_c2: float
    angular: np.ndarray = np.array([])
    if c2_spheroidal.real >= 0:
        sqrt_c2 = math.sqrt(c2_spheroidal.real)
        angular, _ = pro_ang1(
            M_ORDER, n_degree, sqrt_c2, np.cos(LIN_THETA))
    elif c2_spheroidal.real < 0:
        sqrt_c2 = math.sqrt(-c2_spheroidal.real)
        angular, _ = obl_ang1(
            M_ORDER, n_degree, sqrt_c2, np.cos(LIN_THETA))

    fmreigf: np.ndarray = angular / np.sqrt(critical.real)

    sign: int = adjust_sign(fmreigf, NUM_THETA)
    fmreigf *= sign

    return eig, fmreigf, c2_spheroidal.real
#


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger: logging.Logger = logging.getLogger(__name__)

    if ALPHA > 0.5:
        logger.warning('ALPHA should be smaller than 0.5')
        sys.exit()

    PNM_NORM: Final[np.ndarray] = load_legendre(M_ORDER, N_T, NUM_THETA)
    results = wrapper_solve_eig(
        M_ORDER, ALPHA, E_ETA, SIZE_SUBMAT, CRITERION_C)

    plt.rcParams['text.usetex'] = True

    wrapper_choose_eigf(results)
#
