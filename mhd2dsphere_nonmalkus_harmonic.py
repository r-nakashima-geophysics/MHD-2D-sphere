"""A code for 2D ideal MHD waves on a rotating sphere under the
non-Malkus field B_phi = B_0 B(theta) sin(theta)

Plots a figure of the values of L^2 in the differential equation
y''+L^2y=0 of the harmonic oscillation.

Parameters
-----
ALPHA : float
    The Lehnert number

Notes
-----
Parameters other than command line arguments are described below.

References
-----
[1] Nakashima & Yoshida (submitted)

Examples
-----
In the below example, ALPHA will be set to the default value.
    python3 mhd2dsphere_nonmalkus_harmonic.py
In the below example, ALPHA will be set to 1.
    python3 mhd2dsphere_nonmalkus_harmonic.py 1

"""

import math
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Final

import matplotlib.pyplot as plt
import numpy as np

from package.input_arg import input_alpha

FUNC_B: Callable[[np.ndarray], np.ndarray]
FUNC_DB: Callable[[np.ndarray], np.ndarray]
FUNC_DDB: Callable[[np.ndarray], np.ndarray]
TEXT_B: str
NAME_B: str

# ========== parameters ==========

# The boolean value to switch whether to use the magnetostrophic
# approximation
SWITCH_MS: Final[bool] = False

# The function B and its derivatives with respect to theta
B_MALKUS: Final[tuple[Callable[[np.ndarray], np.ndarray],
                Callable[[np.ndarray], np.ndarray],
                Callable[[np.ndarray], np.ndarray],
                str, str]] = (
    lambda theta_rad: np.full_like(theta_rad, 1),
    lambda theta_rad: np.full_like(theta_rad, 0),
    lambda theta_rad: np.full_like(theta_rad, 0),
    r'\sin\theta', 'malkus')
B_SINCOS: Final[tuple[Callable[[np.ndarray], np.ndarray],
                Callable[[np.ndarray], np.ndarray],
                Callable[[np.ndarray], np.ndarray],
                str, str]] = (
    lambda theta_rad: np.cos(theta_rad),
    lambda theta_rad: -np.sin(theta_rad),
    lambda theta_rad: -np.cos(theta_rad),
    r'\sin\theta\cos\theta', 'sincos')
B_SIN2COS: Final[tuple[Callable[[np.ndarray], np.ndarray],
                 Callable[[np.ndarray], np.ndarray],
                 Callable[[np.ndarray], np.ndarray],
                 str, str]] = (
    lambda theta_rad: np.sin(theta_rad)*np.cos(theta_rad),
    lambda theta_rad: np.cos(2*theta_rad),
    lambda theta_rad: -2*np.sin(2*theta_rad),
    r'\sin^2\theta\cos\theta', 'sin2cos')
FUNC_B, FUNC_DB, FUNC_DDB, TEXT_B, NAME_B = B_MALKUS

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = input_alpha(0.1)

# The resolution in the theta direction
THETA_INIT: Final[float] = 0
THETA_STEP: Final[float] = 0.01
THETA_END: Final[float] = math.pi

# The range of the angular frequency
LAMBDA_STEP: Final[float] = ALPHA * 0.01
# for B_MALKUS
LAMBDA_INIT: Final[float] = -M_ORDER * ALPHA * 2
LAMBDA_END: Final[float] = M_ORDER * ALPHA * 2
# for B_SINCOS
# LAMBDA_INIT: Final[float] = -M_ORDER * ALPHA
# LAMBDA_END: Final[float] = M_ORDER * ALPHA
# for B_SIN2COS
# LAMBDA_INIT: Final[float] = -M_ORDER * ALPHA / 2
# LAMBDA_END: Final[float] = M_ORDER * ALPHA / 2

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_nonmalkus_harmonic'
NAME_FIG: Final[str] = 'MHD2Dsphere_nonmalkus_harmonic' \
    + f'_{NAME_B}m{M_ORDER}a{ALPHA}'
NAME_FIG_SUFFIX: Final[tuple[str, str]] = ('.png', '_ms.png')
FIG_DPI: Final[int] = 600

# ================================

NUM_LAMBDA: Final[int] = 1 + int((LAMBDA_END-LAMBDA_INIT)/LAMBDA_STEP)
NUM_THETA: Final[int] = 1 + int((THETA_END-THETA_INIT)/THETA_STEP)

LIN_LAMBDA: Final[np.ndarray] \
    = np.linspace(LAMBDA_INIT, LAMBDA_END, NUM_LAMBDA)
LIN_THETA: Final[np.ndarray] \
    = np.linspace(THETA_INIT, THETA_END, NUM_THETA)

GRID_LAMBDA: np.ndarray
GRID_THETA: np.ndarray
GRID_LAMBDA, GRID_THETA = np.meshgrid(LIN_LAMBDA, LIN_THETA)

EPS: Final[float] = 10**(-10)


def plot_l2() -> None:
    """Plots a figure of the values of L^2"""

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
        r'[$B_{0\phi}=B_0' + TEXT_B + r'$] : '
        + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}',
        color='magenta', fontsize=16)

    fig.tight_layout()

    fig.subplots_adjust(right=0.79, wspace=0.25)
    axpos = axis.get_position()
    cbar_ax: plt.Axes \
        = fig.add_axes([0.81, axpos.y0, 0.01, axpos.height])

    cbar = fig.colorbar(cfs, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label=r'$\mathcal{L}^2$', size=16)

    os.makedirs(PATH_DIR_FIG, exist_ok=True)

    if not SWITCH_MS:
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX[0]
    else:
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX[1]
    #

    path_fig: Path = PATH_DIR_FIG / name_fig_full
    fig.savefig(str(path_fig), dpi=FIG_DPI)
#


def calc_l2() -> np.ndarray:
    """Calculates the values of L^2

    Returns
    -----
    grid_l2 : ndarray
        The values of L^2

    Notes
    -----
    This function is based on eq. (32b) in Nakashima & Yoshida (in
    prep.)[1]_.

    """

    sin: np.ndarray = np.sin(GRID_THETA)
    cos: np.ndarray = np.cos(GRID_THETA)
    critical: np.ndarray \
        = -(M_ORDER**2) * (ALPHA**2) * (FUNC_B(GRID_THETA)**2)

    if not SWITCH_MS:
        critical += (GRID_LAMBDA**2)
    #

    d_critical: np.ndarray = (M_ORDER**2) * (ALPHA**2) \
        * 2 * FUNC_B(GRID_THETA) * FUNC_DB(GRID_THETA) / (sin+EPS)
    dd_critical: np.ndarray = 2 * (M_ORDER**2) * (ALPHA**2) * (
        cos*FUNC_B(GRID_THETA)*FUNC_DB(GRID_THETA)/(sin+EPS)
        - FUNC_B(GRID_THETA)*FUNC_DDB(GRID_THETA)
        - (FUNC_DB(GRID_THETA)**2)) / ((sin**2)+EPS)

    grid_l2: np.ndarray = -(M_ORDER**2) - M_ORDER*(sin**2)*(
        GRID_LAMBDA+2*M_ORDER*(ALPHA**2)*(
            (FUNC_B(GRID_THETA)**2)
            - cos*FUNC_B(GRID_THETA)*FUNC_DB(GRID_THETA)/(sin+EPS))) \
        / (critical+EPS) + (sin**2)*(
        cos*d_critical/(critical+EPS)
        + (sin**2)*(d_critical**2)/(4*(critical**2)+EPS)
        - (sin**2)*dd_critical/(2*critical+EPS))

    return grid_l2
#


def cmap_range(grid_l2: np.ndarray) -> float:
    """A function to determine the range of the color map

    Parameters
    -----
    grid_l2 : ndarray
        The values of L^2

    Returns
    -----
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
