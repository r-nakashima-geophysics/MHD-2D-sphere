"""A code for 2D ideal MHD waves on a rotating sphere under the
non-Malkus field B_phi = B_0 sin(theta) cos(theta)

Plots a figure of the comparison between the Frobenius series solutions
and a numerical solution.

Raises
----------
Lack of fitting data. You need to reduce the value of the variable
NUM_DATA.
    If there are not enough grid points for fitting because a critical
    latitude is near the poles.

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

import caffeine
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from package_common.decorator_yesno import exe_yes_continue
from package_mhd2dsphere.make_eigf import choose_eigf, make_eigf
from package_mhd2dsphere.make_frobenius import (calc_frobenius,
                                                make_fitting_data)
from package_mhd2dsphere.make_legendre import load_legendre
from package_mhd2dsphere.solve_eig import wrapper_solve_eig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== Parameters ==========

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = 0.1

# The truncation degree
N_T: Final[int] = 2000

# The number of the grid in the theta direction
NUM_THETA: Final[int] = 7201
# The number of data points to which we will fit
NUM_DATA: Final[int] = 200

# A criterion for convergence
# degree
N_C: Final[int] = int(N_T/2)
# ratio
R_C: Final[float] = 100

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_sincos_frobenius'
NAME_FIG: Final[str] = 'MHD2Dsphere_sincos_frobenius' \
    + f'_m{M_ORDER}a{ALPHA}N{N_T}th{NUM_THETA}d{NUM_DATA}'
NAME_FIG_SUFFIX: Final[str] = '.png'
FIG_DPI: Final[int] = 600

# ================================

# The magnetic Ekman number
E_ETA: Final[float] = 0

CRITERION_C: Final[tuple[int, float]] = (N_C, R_C)

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

LIN_THETA: Final[np.ndarray] = np.linspace(0, math.pi, NUM_THETA)
LIN_MU: Final[np.ndarray] = np.cos(LIN_THETA)


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

    wrapper_plot_frobenius(psi_vec, vpa_vec, eig, i_chosen)

    plt.show()
#


def wrapper_plot_frobenius(psi_vec: np.ndarray,
                           vpa_vec: np.ndarray,
                           eig: complex,
                           i_mode: int) -> None:
    """A wrapper of a function to plot a figure of the comparison
    between the Frobenius series solutions and a numerical solution

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

    axes: np.ndarray
    theta_c: float
    amp_max: float
    amp_min: float
    (fig, axes), theta_c, [amp_min, amp_max] \
        = plot_frobenius(psi_vec, vpa_vec, eig)

    axes[0].grid()
    axes[1].grid()
    axes[2].grid()
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    axes[2].set_axisbelow(True)

    ymin: float
    ymax: float

    ymin, ymax = axes[0].get_ylim()
    axes[0].plot([0, 0], [ymin, ymax], color='gray', linestyle='-')
    axes[0].set_ylim(ymin, ymax)
    ymin, ymax = axes[1].get_ylim()
    axes[1].plot([0, 0], [ymin, ymax], color='gray', linestyle='-')
    axes[1].set_ylim(ymin, ymax)

    axes[2].set_xlim(0, math.pi/2)
    plt.xticks(
        [0, math.pi/12, math.pi/6, math.pi/4,
            math.pi/3, 5*math.pi/12, math.pi/2],
        ['$0$', '$15$', '$30$', '$45$', '$60$', '$75$', '$90$'])
    axes[2].set_ylim(amp_min, amp_max)

    axes[0].set_xlabel(
        r'$\tilde{\psi}_\mathrm{I\!I}^{(\mathrm{c})}'
        + r'/\tilde{\psi}_\mathrm{I}^{(\mathrm{c})}$', fontsize=16)
    axes[0].set_ylabel(
        r'$\tilde{\psi}_\mathrm{num}'
        + r'/\tilde{\psi}_\mathrm{I}^{(\mathrm{c})}$', fontsize=16)
    axes[1].set_xlabel(
        r'$\tilde{\psi}_\mathrm{I}^{(\mathrm{c})}'
        + r'/\tilde{\psi}_\mathrm{I\!I}^{(\mathrm{c})}$', fontsize=16)
    axes[1].set_ylabel(
        r'$\tilde{\psi}_\mathrm{num}'
        + r'/\tilde{\psi}_\mathrm{I\!I}^{(\mathrm{c})}$', fontsize=16)
    axes[2].set_xlabel(
        'colatitude [degree] (northern hemisphere)', fontsize=16)
    axes[2].set_ylabel('amplitude', fontsize=16)

    axes[0].set_title(
        r'$\tilde{\psi}_\mathrm{num}'
        + r'/\tilde{\psi}_\mathrm{I}^{(\mathrm{c})}'
        + r'=C_\mathrm{I\!I}'
        + r'(\tilde{\psi}_\mathrm{I\!I}^{(\mathrm{c})}'
        + r'/\tilde{\psi}_\mathrm{I}^{(\mathrm{c})})'
        + r'+C_\mathrm{I}$', fontsize=16)
    axes[1].set_title(
        r'$\tilde{\psi}_\mathrm{num}'
        + r'/\tilde{\psi}_\mathrm{I\!I}^{(\mathrm{c})}'
        + r'=C_\mathrm{I}'
        + r'(\tilde{\psi}_\mathrm{I}^{(\mathrm{c})}'
        + r'/\tilde{\psi}_\mathrm{I\!I}^{(\mathrm{c})})'
        + r'+C_\mathrm{I\!I}$', fontsize=16)
    axes[2].set_title(
        r'$\lambda=$' + f' {eig.real:8.5f}, '
        + r'$\theta_\mathrm{c}=$'
        + f' {math.degrees(theta_c.real):4.2f} '
        + r'$\mathrm{[deg]}$', fontsize=16)

    fig.suptitle(
        r'Frobenius series solutions vs Numerical solution '
        + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
        + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}',
        fontsize=16)

    leg: list[plt.Legend] = [axes[0].legend(loc='best', fontsize=13),
                             axes[1].legend(loc='best', fontsize=13),
                             axes[2].legend(loc='best', fontsize=13)]
    leg[0].get_frame().set_alpha(1)
    leg[1].get_frame().set_alpha(1)
    leg[2].get_frame().set_alpha(1)

    axes[0].tick_params(labelsize=14)
    axes[1].tick_params(labelsize=14)
    axes[2].tick_params(labelsize=14)
    axes[0].minorticks_on()
    axes[1].minorticks_on()
    axes[2].minorticks_on()

    fig.tight_layout()

    name_fig_full: str = NAME_FIG + f'_{i_mode+1}' + NAME_FIG_SUFFIX

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / name_fig_full
    fig.savefig(path_fig, dpi=FIG_DPI)
#


def plot_frobenius(psi_vec: np.ndarray,
                   vpa_vec: np.ndarray,
                   eig: complex) -> tuple[tuple, float, list[float]]:
    """Plots a figure of the comparison between a Frobenius series
    solution and a numerical solution

    Parameters
    ----------
    psi_vec : ndarray
        An eigenvector of the stream function (psi)
    vpa_vec : ndarray
        An eigenvector of the vector potential (a)
    eig : complex
        An eigenvalue

    Returns
    ----------
    fig_bundle : tuple
        A tuple of figures
    theta_c : float
        A critical colatitude
    [amp_min, amp_max] : list of float
        The y limits of a graph

    """

    mu_c_tmp: complex = eig / (M_ORDER * ALPHA)
    mu_c: complex
    if 0 <= np.arccos(mu_c_tmp).real < math.pi/2:
        mu_c = mu_c_tmp
    else:
        mu_c = -mu_c_tmp

    theta_c: float = np.arccos(mu_c).real
    i_theta_c: int = int(np.argmin(np.abs(LIN_THETA-theta_c)))

    if (i_theta_c < NUM_DATA) or (NUM_THETA < i_theta_c):
        logger.warning(
            'Lack of fitting data. '
            + 'You need to reduce the value of the variable NUM_DATA.')
        sys.exit()

    psi: np.ndarray = np.array([])
    psi1: np.ndarray = np.array([])
    psi2: np.ndarray = np.array([])

    psi, _ = make_eigf(psi_vec, vpa_vec, M_ORDER, PNM_NORM)
    psi1, psi2 = calc_frobenius(M_ORDER, ALPHA, NUM_THETA, eig, mu_c)

    fig = plt.figure(figsize=(10, 10))
    gspec = GridSpec(5, 2)
    sspec: list = [
        gspec.new_subplotspec((0, 0), rowspan=3),
        gspec.new_subplotspec((0, 1), rowspan=3),
        gspec.new_subplotspec((3, 0), rowspan=2, colspan=2)
    ]
    axes: list[plt.Axes] = [
        fig.add_subplot(sspec[0]),
        fig.add_subplot(sspec[1]),
        fig.add_subplot(sspec[2]),
    ]

    a1_eq: float
    b1_eq: float
    a1_pole: float
    b1_pole: float
    a2_eq: float
    b2_eq: float
    a2_pole: float
    b2_pole: float

    y_eq: np.ndarray \
        = make_fitting_data(psi, psi1, NUM_DATA, i_theta_c, 'eq')
    x_eq: np.ndarray \
        = make_fitting_data(psi2, psi1, NUM_DATA, i_theta_c, 'eq')
    [a1_eq, b1_eq] = np.polyfit(x_eq, y_eq, 1)
    y_pole: np.ndarray \
        = make_fitting_data(psi, psi1, NUM_DATA, i_theta_c, 'pole')
    x_pole: np.ndarray \
        = make_fitting_data(psi2, psi1, NUM_DATA, i_theta_c, 'pole')
    [a1_pole, b1_pole] = np.polyfit(x_pole, y_pole, 1)

    axes[0].scatter(x_pole, y_pole, s=10, color='blue',
                    label=r'$\mu>\mu_\mathrm{c}$')
    axes[0].scatter(x_eq, y_eq, s=10, color='red',
                    label=r'$\mu<\mu_\mathrm{c}$')

    y_eq = make_fitting_data(psi, psi2, NUM_DATA, i_theta_c, 'eq')
    x_eq = make_fitting_data(psi1, psi2, NUM_DATA, i_theta_c, 'eq')
    [a2_eq, b2_eq] = np.polyfit(x_eq, y_eq, 1)
    y_pole = make_fitting_data(psi, psi2, NUM_DATA, i_theta_c, 'pole')
    x_pole = make_fitting_data(psi1, psi2, NUM_DATA, i_theta_c, 'pole')
    [a2_pole, b2_pole] = np.polyfit(x_pole, y_pole, 1)

    axes[1].scatter(x_pole, y_pole, s=10, color='blue',
                    label=r'$\mu>\mu_\mathrm{c}$')
    axes[1].scatter(x_eq, y_eq, s=10, color='red',
                    label=r'$\mu<\mu_\mathrm{c}$')

    lin_x: list[np.ndarray] = [np.array([]), ] * 2
    xmin: float
    xmax: float

    xmin, xmax = axes[0].get_xlim()
    lin_x[0] = np.linspace(min(xmin, 0), max(xmax, 0), 100)
    xmin, xmax = axes[1].get_xlim()
    lin_x[1] = np.linspace(min(xmin, 0), max(xmax, 0), 100)

    axes[0].plot(
        lin_x[0], a1_pole*lin_x[0] + b1_pole, color='black',
        linestyle='--', label=r'$y=$' + f' {a1_pole:>8.5f} '
        + r'$x$' + f' {b1_pole:>+8.5f}')
    axes[0].plot(
        lin_x[0], a1_eq*lin_x[0] + b1_eq, color='black',
        linestyle='-', label=r'$y=$' + f' {a1_eq:>8.5f} '
        + r'$x$' + f' {b1_eq:>+8.5f}')
    axes[1].plot(
        lin_x[1], a2_pole*lin_x[1] + b2_pole, color='black',
        linestyle='--', label=r'$y=$' + f' {a2_pole:>8.5f} '
        + r'$x$' + f' {b2_pole:>+8.5f}')
    axes[1].plot(
        lin_x[1], a2_eq*lin_x[1] + b2_eq, color='black',
        linestyle='-', label=r'$y=$' + f' {a2_eq:>8.5f} '
        + r'$x$' + f' {b2_eq:>+8.5f}')

    amp_max: float = 1.1 * np.nanmax(np.abs(psi.real))
    amp_min: float = -amp_max

    fit_range: np.ndarray = np.linspace(
        LIN_THETA[i_theta_c-NUM_DATA],
        LIN_THETA[i_theta_c+NUM_DATA], 2*NUM_DATA)
    axes[2].fill_between(
        fit_range, amp_max, amp_min, facecolor='lightgray')

    c_1_eq: float = (b1_eq + a2_eq) / 2
    c_1_pole: float = (b1_pole + a2_pole) / 2
    c_2: float = (a1_pole + a1_eq + b2_pole + b2_eq) / 4

    psi_sum_eq: np.ndarray \
        = c_1_eq * psi1.real + c_2 * psi2.real
    psi_sum_pole: np.ndarray \
        = c_1_pole * psi1.real + c_2 * psi2.real

    lin_mu_eq: np.ndarray = np.full(NUM_THETA, np.nan)
    lin_mu_pole: np.ndarray = np.full(NUM_THETA, np.nan)
    for i_theta in range(NUM_THETA):
        if i_theta < i_theta_c:
            lin_mu_pole[i_theta] = 1
        elif i_theta > i_theta_c:
            lin_mu_eq[i_theta] = 1

    axes[2].plot(
        LIN_THETA, psi.real, color='black',
        label='stream function ' + r'$\tilde{\psi}_\mathrm{num}$')
    axes[2].plot(
        LIN_THETA, psi_sum_pole * lin_mu_pole, color='blue',
        label=f' {c_1_pole:>8.5f} '
        + r'$\tilde{\psi}_\mathrm{I}^{(\mathrm{c})}$'
        + f' {c_2:>+8.5f} '
        + r'$\tilde{\psi}_\mathrm{I\!I}^{(\mathrm{c})}$')
    axes[2].plot(
        LIN_THETA, psi_sum_eq * lin_mu_eq, color='red',
        label=f' {c_1_eq:>8.5f} '
        + r'$\tilde{\psi}_\mathrm{I}^{(\mathrm{c})}$'
        + f' {c_2:>+8.5f} '
        + r'$\tilde{\psi}_\mathrm{I\!I}^{(\mathrm{c})}$')

    fig_bundle: tuple = (fig, axes)

    return fig_bundle, theta_c, [amp_min, amp_max]
#


if __name__ == '__main__':

    caffeine.on(display=False)

    PNM_NORM: Final[np.ndarray] \
        = load_legendre(M_ORDER, N_T, NUM_THETA)
    results: tuple[np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray] \
        = wrapper_solve_eig(
        M_ORDER, ALPHA, E_ETA, SIZE_SUBMAT, CRITERION_C)

    plt.rcParams['text.usetex'] = True

    wrapper_choose_eigf(results)
#
