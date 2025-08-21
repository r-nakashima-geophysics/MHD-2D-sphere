"""A code for 2D ideal MHD waves on a rotating sphere under the
non-Malkus field B_phi = B_0 sin(theta) cos(theta)

Plots a figure of the approximate dispersion relation for fast magnetic
Rossby (MR) waves.

Parameters
----------
M_ORDER : int
    The zonal wavenumber (order)

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

Examples
----------
In the below example, M_ORDER will be set to the default value.
    python3 mhd2dsphere_sincos_fmr.py
In the below example, M_ORDER will be set to 2.
    python3 mhd2dsphere_sincos_fmr.py 2
"""

import math
import os
from pathlib import Path
from time import perf_counter
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from package_common.utils_input import input_value
from package_mhd2dsphere.dispersion_fmr import dispersion_fmr
from package_mhd2dsphere.load_data import wrapper_load_results
from package_mhd2dsphere.processing_results import pickup_eig

# ========== Parameters ==========

# The variable to switch approximate dispersion relations
# spheroidal wave function ('sph')
# wkbj quantization ('wkb')
SWITCH_EQ: Final[str] = 'wkb'

# The zonal wavenumber (order)
M_ORDER: Final[int] = input_value(1, int)

# Degrees
N_INIT: Final[int] = M_ORDER  # M_ORDER <= N_INIT
N_STEP: Final[int] = 1
N_END: Final[int] = 20

# The truncation degree
N_T: Final[int] = 2000

# The range of eigenvalues
# real part
EIG_RE_LOG_INIT: Final[float] = -3
EIG_RE_LOG_END: Final[float] = 0

# The range of the Lehnert number
ALPHA_LOG_INIT: Final[float] = -3
ALPHA_LOG_END: Final[float] = 0

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_sincos_fmr'
NAME_FIG: Final[str] \
    = f'MHD2Dsphere_sincos_fmr_m{M_ORDER}N{N_T}_{SWITCH_EQ}.png'
FIG_DPI: Final[int] = 600

# ================================

# The magnetic Ekman number
E_ETA: Final[float] = 0

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

NUM_N: Final[int] = 1 + int((N_END-N_INIT)/N_STEP)
LIN_N: Final[np.ndarray] = np.linspace(N_INIT, N_END, NUM_N)


def wrapper_plot_fmr(
        bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray]) -> None:
    """A wrapper of a function to plot a figure of the approximate
    dispersion relation for fast magnetic Rossby (MR) waves

    Parameters
    ----------
    bundle : tuple of ndarray
        A tuple of results

    """

    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plot_fmr(bundle)

    axes[0].grid()
    axes[1].grid()

    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)

    axes[0].set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    axes[1].set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    axes[0].set_ylim(10**EIG_RE_LOG_INIT, 10**EIG_RE_LOG_END)
    axes[1].set_ylim(10**EIG_RE_LOG_INIT, 10**EIG_RE_LOG_END)

    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')

    axes[0].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    axes[1].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    axes[0].set_ylabel(
        r'$|\lambda|=|\omega/2\Omega_0|$',
        fontsize=16)

    axes[0].set_title(
        r'Sinuous, Retrograde ($\lambda<0$)',
        fontsize=16)
    axes[1].set_title(
        r'Varicose, Retrograde ($\lambda<0$)',
        fontsize=16)

    fig.suptitle(
        r'Dispersion relation [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
        + r'$m=$' + f' {M_ORDER}', fontsize=16)

    leg1: plt.Legend = axes[0].legend(loc='lower right', fontsize=14)
    leg2: plt.Legend = axes[1].legend(loc='lower right', fontsize=14)
    leg1.get_frame().set_alpha(1)
    leg2.get_frame().set_alpha(1)

    axes[0].tick_params(labelsize=14)
    axes[1].tick_params(labelsize=14)
    axes[0].minorticks_on()
    axes[1].minorticks_on()

    fig.tight_layout()

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / NAME_FIG
    fig.savefig(path_fig, dpi=FIG_DPI)
#


def plot_fmr(
    bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray, np.ndarray]) \
        -> tuple[plt.Figure, np.ndarray]:
    """Plots a figure of the approximate dispersion relation for fast
    magnetic Rossby (MR) waves

    Parameters
    ----------
    bundle : tuple of ndarray
        A tuple of results

    Returns
    ----------
    fig_bundle : tuple
        A tuple of figures

    """

    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    lin_alpha: np.ndarray
    eig: np.ndarray
    mke: np.ndarray
    sym: np.ndarray
    lin_alpha, eig, mke, _, _, sym = bundle

    num_alpha: int = len(lin_alpha)

    alpha: float
    ones_alpha: np.ndarray

    dict_eig: dict[str, np.ndarray]

    for i_alpha in range(num_alpha):
        alpha = 10**lin_alpha[i_alpha]

        ones_alpha = np.full(SIZE_MAT, alpha)

        dict_eig = pickup_eig(
            eig[i_alpha, :], mke[i_alpha, :], sym[i_alpha, :])

        dict_eig['sr'] = -np.conjugate(dict_eig['sr'])
        dict_eig['vr'] = -np.conjugate(dict_eig['vr'])

        axes[0].scatter(ones_alpha, dict_eig['sr'].real, s=0.5, c='gray')
        axes[1].scatter(ones_alpha, dict_eig['vr'].real, s=0.5, c='gray')

    alpha_tmp: np.ndarray = np.zeros(num_alpha)
    for i_alpha in range(0, num_alpha, int(num_alpha/60)):
        alpha_tmp[i_alpha] = 10**lin_alpha[i_alpha]

    lin_alpha_skip: np.ndarray = alpha_tmp[np.nonzero(alpha_tmp)]
    num_alpha_skip: int = len(lin_alpha_skip)
    ones_alpha_skip: np.ndarray = np.zeros((NUM_N, num_alpha_skip))

    start: float = float()
    n_degree: int

    sol: float
    sol_s: np.ndarray = np.full((NUM_N, num_alpha_skip), math.nan)
    sol_v: np.ndarray = np.full((NUM_N, num_alpha_skip), math.nan)

    for i_alpha in range(num_alpha_skip):
        alpha = lin_alpha_skip[i_alpha]

        ones_alpha_skip[:, i_alpha] = np.full(NUM_N, alpha)

        for i_n in range(NUM_N):
            n_degree = LIN_N[i_n]

            if SWITCH_EQ == 'sph':
                start = -10**(-10)
            elif SWITCH_EQ == 'wkb':
                start = -(1+10**(-5))*M_ORDER*alpha

            sol = root(dispersion_fmr, start,
                       args=(M_ORDER, n_degree, alpha, SWITCH_EQ),
                       method='hybr').x[0].real

            if ((M_ORDER*alpha)**2)/(sol**2) < 0.99:
                if i_n % 2 == 0:
                    sol_s[i_n, i_alpha] = -1 * sol
                    sol_v[i_n, i_alpha] = math.nan
                elif i_n % 2 == 1:
                    sol_s[i_n, i_alpha] = math.nan
                    sol_v[i_n, i_alpha] = -1 * sol

    axes[0].scatter(ones_alpha_skip, sol_s, s=10, c='red',
                    label=r'$\lambda_\mathrm{approx}$')
    axes[1].scatter(ones_alpha_skip, sol_v, s=10, c='red',
                    label=r'$\lambda_\mathrm{approx}$')

    fig_bundle: tuple[plt.Figure, np.ndarray] = (fig, axes)

    return fig_bundle
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    results, results_log \
        = wrapper_load_results((False, True), M_ORDER, E_ETA, N_T)

    plt.rcParams['text.usetex'] = True

    wrapper_plot_fmr(results_log)

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
