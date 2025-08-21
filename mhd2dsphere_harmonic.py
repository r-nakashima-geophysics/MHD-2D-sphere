"""A code for 2D ideal MHD waves on a rotating sphere under the
non-Malkus field B_phi = B_0 B(theta) sin(theta)

Plots a figure of the values of L^2 in the differential equation
y''+L^2y=0.

Parameters
----------
ALPHA : float
    The Lehnert number

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
In the below example, ALPHA will be set to the default value.
    python3 mhd2dsphere_harmonic.py
In the below example, ALPHA will be set to 1.
    python3 mhd2dsphere_harmonic.py 1
"""

import math
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Final

import matplotlib.pyplot as plt
import numpy as np

from package_common.utils_input import input_value
from package_mhd2dsphere import func_b, func_u

FUNC_B: Callable[[complex], complex]
FUNC_DB: Callable[[complex], complex]
FUNC_D2B: Callable[[complex], complex]
TEX_B: str
NAME_B: str


# ========== Parameters ==========

# The boolean value to switch whether to use the magnetostrophic
# approximation or not
SWITCH_MS: Final[bool] = False

# The function B
# FUNC_B, FUNC_DB, FUNC_D2B, TEX_B, NAME_B = func_b.b_hydro('theta')
FUNC_B, FUNC_DB, FUNC_D2B, TEX_B, NAME_B = func_b.b_malkus('theta')
# FUNC_B, FUNC_DB, FUNC_D2B, TEX_B, NAME_B = func_b.b_sincos('theta')
# FUNC_B, FUNC_DB, FUNC_D2B, TEX_B, NAME_B = func_b.b_sin2cos('theta')

# The function U
FUNC_U, FUNC_DU, FUNC_D2U, TEX_U, NAME_U = func_u.u_rigid('theta')

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = input_value(0.1, float)

# The number of the grid in the theta direction
THETA_INIT: Final[float] = 0
THETA_STEP: Final[float] = 0.01
THETA_END: Final[float] = math.pi

# The range of the angular frequency
LAMBDA_STEP: Final[float] = ALPHA * 0.01
# for b_malkus
LAMBDA_INIT: Final[float] = -M_ORDER * ALPHA * 2
LAMBDA_END: Final[float] = M_ORDER * ALPHA * 2
# for b_sincos
# LAMBDA_INIT: Final[float] = -M_ORDER * ALPHA
# LAMBDA_END: Final[float] = M_ORDER * ALPHA
# for b_sin2cos
# LAMBDA_INIT: Final[float] = -M_ORDER * ALPHA / 2
# LAMBDA_END: Final[float] = M_ORDER * ALPHA / 2

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_harmonic'
NAME_FIG: Final[str] = 'MHD2Dsphere_harmonic' \
    + f'_{NAME_B}_m{M_ORDER}a{ALPHA}'
NAME_FIG_SUFFIX: Final[tuple[str, str]] = ('.png', '_ms.png')
FIG_DPI: Final[int] = 600

# ================================

NUM_LAMBDA: Final[int] = 1 + int((LAMBDA_END-LAMBDA_INIT)/LAMBDA_STEP)
NUM_THETA: Final[int] = 1 + int((THETA_END-THETA_INIT)/THETA_STEP)

LIN_LAMBDA: Final[np.ndarray] \
    = np.linspace(LAMBDA_INIT, LAMBDA_END, NUM_LAMBDA)
LIN_THETA: Final[np.ndarray] \
    = np.linspace(THETA_INIT, THETA_END, NUM_THETA)

LIN_B: Final[np.ndarray] = np.array(
    [FUNC_B(LIN_THETA[i_theta]).real for i_theta in range(NUM_THETA)])
LIN_DB: Final[np.ndarray] = np.array(
    [FUNC_DB(LIN_THETA[i_theta]).real for i_theta in range(NUM_THETA)])
LIN_D2B: Final[np.ndarray] = np.array(
    [FUNC_D2B(LIN_THETA[i_theta]).real for i_theta in range(NUM_THETA)])

GRID_LAMBDA: np.ndarray
GRID_THETA: np.ndarray
GRID_B: np.ndarray
GRID_DB: np.ndarray
GRID_D2B: np.ndarray
GRID_LAMBDA, GRID_THETA = np.meshgrid(LIN_LAMBDA, LIN_THETA)
_, GRID_B = np.meshgrid(LIN_LAMBDA, LIN_B)
_, GRID_DB = np.meshgrid(LIN_LAMBDA, LIN_DB)
_, GRID_D2B = np.meshgrid(LIN_LAMBDA, LIN_D2B)

EPS: Final[float] = 10**(-10)


def plot_l2() -> None:
    """Plots a figure of the values of L^2"""

    fig: plt.Figure
    axis: plt.Axes
    fig, axis = plt.subplots(figsize=(5, 5))

    grid_l2: np.ndarray = calc_l2()

    cmap_max: float = cmap_range(grid_l2)
    level_l2: np.ndarray \
        = np.arange(-cmap_max, 1.1*cmap_max, 0.2*cmap_max)

    cfs = axis.contourf(
        GRID_LAMBDA, GRID_THETA, grid_l2, levels=level_l2,
        vmin=-cmap_max, vmax=cmap_max, cmap='bwr_r', extend='both')
    axis.contour(
        GRID_LAMBDA, GRID_THETA, grid_l2, levels=[-1, 0],
        colors='black', linewidths=1)

    axis.grid(linestyle=':')

    axis.set_xlim(LAMBDA_INIT, LAMBDA_END)
    axis.set_ylim(THETA_END, THETA_INIT)

    axis.set_yticks([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi])
    axis.set_yticklabels(['$0$', '$45$', '$90$', '$135$', '$180$'])

    axis.set_xlabel(
        r'$\lambda=\omega/2\Omega_0$', fontsize=16)
    axis.set_ylabel('colatitude [degree]', fontsize=16)

    axis.tick_params(labelsize=14)
    axis.minorticks_on()

    fig.suptitle(
        r'[$B_{0\phi}=' + TEX_B + r'$] : '
        + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}',
        fontsize=16)

    fig.tight_layout()

    fig.subplots_adjust(right=0.79, wspace=0.25)
    axpos = axis.get_position()
    cbar_ax: plt.Axes \
        = fig.add_axes((0.81, axpos.y0, 0.01, axpos.height))

    cbar = fig.colorbar(cfs, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label=r'$\mathcal{L}^2$', size=16)

    os.makedirs(PATH_DIR_FIG, exist_ok=True)

    if not SWITCH_MS:
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX[0]
    else:
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX[1]

    path_fig: Path = PATH_DIR_FIG / name_fig_full
    fig.savefig(path_fig, dpi=FIG_DPI)


def calc_l2() -> np.ndarray:
    """Calculates the values of L^2

    Returns
    ----------
    grid_l2 : ndarray
        The values of L^2

    Notes
    ----------
    This function is based on eq. (25a) in Nakashima & Yoshida
    (2024)[1]_.
    """

    sin: np.ndarray = np.sin(GRID_THETA)
    cos: np.ndarray = np.cos(GRID_THETA)
    critical: np.ndarray = -(M_ORDER**2) * (ALPHA**2) * (GRID_B**2)

    if not SWITCH_MS:
        critical += (GRID_LAMBDA**2)

    d_critical: np.ndarray = (M_ORDER**2) * (ALPHA**2) \
        * 2 * GRID_B * GRID_DB / (sin+EPS)
    dd_critical: np.ndarray \
        = 2 * (M_ORDER**2) * (ALPHA**2) * (
            cos*GRID_B*GRID_DB/(sin+EPS)
            - GRID_B*GRID_D2B - (GRID_DB**2)
        ) / ((sin**2)+EPS)

    grid_l2: np.ndarray \
        = -(M_ORDER**2) - M_ORDER*(sin**2)*(
            GRID_LAMBDA + 2*M_ORDER*(ALPHA**2)
            * ((GRID_B**2)-cos*GRID_B*GRID_DB/(sin+EPS))
        ) / (critical+EPS) + (sin**2)*(
            cos*d_critical/(critical+EPS)
            + (sin**2)*(d_critical**2)/(4*(critical**2)+EPS)
            - (sin**2)*dd_critical/(2*critical+EPS)
        )

    return grid_l2
#


def cmap_range(grid_l2: np.ndarray) -> float:
    """Determines the range of the color map

    Parameters
    ----------
    grid_l2 : ndarray
        The values of L^2

    Returns
    ----------
    cmap_max : float
        The maximum value of the color map

    """

    i_almost_max: int = int(NUM_LAMBDA * NUM_THETA * 0.9)

    cmap_max: float \
        = sorted(np.abs(np.ravel(grid_l2)))[i_almost_max]

    return cmap_max
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    plt.rcParams['text.usetex'] = True

    plot_l2()

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
