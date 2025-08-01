"""A code for 2D ideal MHD waves on a rotating sphere under the
non-Malkus field B_phi = B_0 B(theta) sin(theta)

Plots 2 figures (k-l, k-lambda) of the local dispersion relation.

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

import math
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Final

import matplotlib.pyplot as plt
import numpy as np

from package_mhd2dsphere import func_b, func_u

FUNC_B: Callable[[complex], complex]
TEX_B: str
NAME_B: str

# ========== Parameters ==========

# The boolean value to switch whether to use the magnetostrophic
# approximation
SWITCH_MS: Final[bool] = False

# The function B
# FUNC_B, _, _, TEX_B, NAME_B = func_b.b_hydro('theta')
FUNC_B, _, _, TEX_B, NAME_B = func_b.b_malkus('theta')
# FUNC_B, _, _, TEX_B, NAME_B = func_b.b_sincos('theta')
# FUNC_B, _, _, TEX_B, NAME_B = func_b.b_sin2cos('theta')

# The function U
FUNC_U, _, _, TEX_U, NAME_U = func_u.u_rigid('theta')

# The range of the colatitude
THETA_INIT: Final[float] = 30
THETA_STEP: Final[float] = 30
THETA_END: Final[float] = 90

# The range of the local zonal wavenumber
K_INIT_1: Final[float] = -5
K_INIT_2: Final[float] = 0
K_STEP: Final[float] = 0.01
K_END: Final[float] = 5

# The range of the local latitudinal wavenumber
L_INIT_1: Final[float] = -5
L_INIT_2: Final[float] = -5**2
L_STEP: Final[float] = 0.01
L_END_1: Final[float] = 5
L_END_2: Final[float] = 5**2

# The range of the scaled angular frequency
LAMBDA_INIT_1: Final[float] = 0
LAMBDA_INIT_2: Final[float] = -3
LAMBDA_STEP: Final[float] = 0.01
LAMBDA_END: Final[float] = 3

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_local'
NAME_FIG: Final[str] = f'MHD2Dsphere_local_{NAME_B}'
NAME_FIG_SUFFIX_1: Final[tuple[str, str]] = ('_kl.png', '_kl_ms.png')
NAME_FIG_SUFFIX_2: Final[tuple[str, str]] \
    = ('_klambda.png', '_klambda_ms.png')
FIG_DPI: Final[int] = 600

# ================================

NUM_K_1: Final[int] = 1 + int((K_END-K_INIT_1)/K_STEP)
NUM_K_2: Final[int] = 1 + int((K_END-K_INIT_2)/K_STEP)
NUM_L: Final[int] = 1 + int((L_END_1-L_INIT_1)/L_STEP)
NUM_LAMBDA: Final[int] = 1 + int((LAMBDA_END-LAMBDA_INIT_2)/LAMBDA_STEP)
NUM_THETA: Final[int] = 1 + int((THETA_END-THETA_INIT)/THETA_STEP)

LIN_THETA: Final[np.ndarray] \
    = np.linspace(THETA_INIT, THETA_END, NUM_THETA)

LIN_K_1: Final[np.ndarray] = np.linspace(K_INIT_1, K_END, NUM_K_1)
LIN_K_2: Final[np.ndarray] = np.linspace(K_INIT_2, K_END, NUM_K_2)
LIN_L: Final[np.ndarray] = np.linspace(L_INIT_1, L_END_1, NUM_L)
LIN_LAMBDA: Final[np.ndarray] \
    = np.linspace(LAMBDA_INIT_2, LAMBDA_END, NUM_LAMBDA)

GRID_K_1: np.ndarray
GRID_L: np.ndarray
GRID_K_2: np.ndarray
GRID_LAMBDA: np.ndarray
GRID_K_1, GRID_L = np.meshgrid(LIN_K_1, LIN_L)
GRID_K_2, GRID_LAMBDA = np.meshgrid(LIN_K_2, LIN_LAMBDA)

NUM_COL: Final[int] = min(NUM_THETA, 3)
NUM_ROW: Final[int] = math.ceil(NUM_THETA/NUM_COL)

LEVEL_LAMBDA: Final[np.ndarray] \
    = np.arange(
    LAMBDA_INIT_1, 1.1*LAMBDA_END, 0.1*(LAMBDA_END-LAMBDA_INIT_1))
LEVEL_L2: Final[np.ndarray] \
    = np.arange(L_INIT_2, 1.1*L_END_2, 0.1*(L_END_2-L_INIT_2))

EPS: Final[float] = 10**(-10)


def plot_kl() -> None:
    """Plots a figure of the local dispersion relation (k-l)"""

    fig: plt.Figure = plt.figure(figsize=(5*NUM_COL, 5*NUM_ROW))
    axis: plt.Axes

    theta_deg: float
    theta_rad: float
    g_lambda: np.ndarray

    for i_theta in range(NUM_THETA):

        axis = fig.add_subplot(NUM_ROW, NUM_COL, i_theta+1)

        theta_deg = LIN_THETA[i_theta]
        theta_rad = math.radians(theta_deg)

        g_lambda = calc_lambda(theta_rad)

        cfs = axis.contourf(
            GRID_K_1, GRID_L, g_lambda, levels=LEVEL_LAMBDA,
            vmin=LAMBDA_INIT_1, vmax=LAMBDA_END, cmap='inferno',
            extend='max')

        axis.grid(linestyle=':')

        axis.set_xlim(K_INIT_1, K_END)
        axis.set_ylim(L_INIT_1, L_END_1)

        axis.set_xlabel(r'$k$', fontsize=18)
        axis.set_ylabel(r'$l$', fontsize=18)

        axis.set_title(
            r'$\theta=$' + f' {theta_deg:2.0f} ' + r'$\mathrm{[deg]}$', fontsize=18)

        axis.tick_params(labelsize=16)
        axis.minorticks_on()

        axis.set_aspect('equal')

    fig.suptitle(
        r'Local dispersion relation [$B_{0\phi}=' + TEX_B + r'$]',
        fontsize=18)

    fig.tight_layout()

    fig.subplots_adjust(right=0.91, wspace=0.25)
    axpos = axis.get_position()
    cbar_ax: plt.Axes \
        = fig.add_axes((0.93, axpos.y0, 0.01, axpos.height))

    cbar = fig.colorbar(cfs, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label=r'$|\alpha|^{-1/2}\lambda$', size=18)

    os.makedirs(PATH_DIR_FIG, exist_ok=True)

    name_fig_full: str
    if not SWITCH_MS:
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX_1[0]
    else:
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX_1[1]

    path_fig: Path = PATH_DIR_FIG / name_fig_full
    fig.savefig(path_fig, dpi=FIG_DPI)
#


def plot_klambda() -> None:
    """Plots a figure of the local dispersion relation (k-lambda)"""

    fig: plt.Figure = plt.figure(figsize=(5*NUM_COL, 7*NUM_ROW))
    axis: plt.Axes

    theta_deg: float
    theta_rad: float
    g_l2: np.ndarray

    for i_theta in range(NUM_THETA):

        axis = fig.add_subplot(NUM_ROW, NUM_COL, i_theta+1)

        theta_deg = LIN_THETA[i_theta]
        theta_rad = math.radians(theta_deg)

        g_l2 = calc_l2(theta_rad)

        cfs = axis.contourf(
            GRID_K_2, GRID_LAMBDA, g_l2, levels=LEVEL_L2,
            vmin=L_INIT_2, vmax=L_END_2, cmap='bwr_r', extend='both')
        axis.contour(
            GRID_K_2, GRID_LAMBDA, g_l2, levels=LEVEL_L2,
            colors='k', linewidths=0.8)

        axis.grid(linestyle=':')

        axis.set_xlim(K_INIT_2, K_END)
        axis.set_ylim(LAMBDA_INIT_2, LAMBDA_END)

        axis.set_xlabel(r'$k$', fontsize=18)
        axis.set_ylabel(r'$|\alpha|^{-1/2}\lambda$', fontsize=18)

        axis.set_title(
            r'$\theta=$' + f' {theta_deg:2.0f} ' + r'$\mathrm{[deg]}$',
            fontsize=18)

        axis.tick_params(labelsize=16)
        axis.minorticks_on()

    fig.suptitle(
        r'Local dispersion relation [$B_{0\phi}=' + TEX_B + r'$]',
        fontsize=18)

    fig.tight_layout()

    fig.subplots_adjust(right=0.91, wspace=0.25)
    axpos = axis.get_position()
    cbar_ax: plt.Axes \
        = fig.add_axes((0.93, axpos.y0, 0.01, axpos.height))

    cbar = fig.colorbar(cfs, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=r'$l^2$', size=18)

    os.makedirs(PATH_DIR_FIG, exist_ok=True)

    name_fig_full: str
    if not SWITCH_MS:
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX_2[0]
    else:
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX_2[1]

    path_fig: Path = PATH_DIR_FIG / name_fig_full
    fig.savefig(path_fig, dpi=FIG_DPI)
#


def calc_lambda(theta_rad: float) -> np.ndarray:
    """Calculates the local dispersion relation

    Parameters
    ----------
    theta_rad : float
        A colatitude

    Returns
    ----------
    g_lambda : ndarray
        Eigenvalues

    Notes
    ----------
    This function is based on eq. (33a) in Nakashima & Yoshida
    (2024)[1]_.

    """

    sin: float = math.sin(theta_rad)
    wavenum2: np.ndarray = (GRID_K_1**2) + (GRID_L**2)
    value_b: float = FUNC_B(theta_rad).real

    rossby: np.ndarray
    sq_rt: np.ndarray
    g_lambda: np.ndarray
    if not SWITCH_MS:
        rossby = -sin * GRID_K_1 / (wavenum2+EPS)
        sq_rt = np.sqrt(
            (rossby**2) + 4*(GRID_K_1**2)*(value_b**2)*(sin**2))
        g_lambda = (rossby+sq_rt) / 2
    else:
        g_lambda = GRID_K_1 * (value_b**2) * sin * wavenum2

    return g_lambda
#


def calc_l2(theta_rad: float) -> np.ndarray:
    """Calculates the local dispersion relation

    Parameters
    ----------
    theta_rad : float
        A colatitude

    Returns
    ----------
    g_l2 : ndarray
        Squared meridional wavenumbers

    Notes
    ----------
    This function is based on eq. (33b) in Nakashima & Yoshida
    (2024)[1]_.

    """

    sin: float = math.sin(theta_rad)
    value_b: float = FUNC_B(theta_rad).real
    critical: np.ndarray \
        = -(GRID_K_2**2) * (value_b**2) * (sin**2)

    if not SWITCH_MS:
        critical += (GRID_LAMBDA**2)

    g_l2: np.ndarray \
        = -(GRID_K_2**2) - GRID_LAMBDA*GRID_K_2*sin/(critical+EPS)

    return g_l2
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    plt.rcParams['text.usetex'] = True

    plot_kl()
    plot_klambda()

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
