"""A code for 2D ideal MHD waves on a rotating sphere under the
non-Malkus field B_phi = B_0 sin(theta) cos(theta)

Plots a figure displaying the discontinuity in the coefficient of the
first Frobenius series solution for all the eigenfunctions.

Parameters
----------
ALPHA : float
    The Lehnert number

Notes
----------
Parameters other than command line arguments are described below.

References
----------
[1] Nakashima & Yoshida (submitted)

"""

import math
import os
from pathlib import Path
from time import perf_counter
from typing import Final

import caffeine
import matplotlib.pyplot as plt
import numpy as np

from package_common.input_arg import input_value
from package_mhd2dsphere.load_data import load_legendre
from package_mhd2dsphere.make_eigf import make_eigf
from package_mhd2dsphere.make_frobenius import (calc_frobenius,
                                                make_fitting_data)
from package_mhd2dsphere.processing_results import sort_sv
from package_mhd2dsphere.solve_eig import wrapper_solve_eig

# ========== Parameters ==========

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = input_value(0.1, float)

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
    = Path('.') / 'fig' / 'MHD2Dsphere_sincos_allfrobenius'
NAME_FIG: Final[str] = 'MHD2Dsphere_sincos_allfrobenius' \
    + f'_m{M_ORDER}a{ALPHA}N{N_T}th{NUM_THETA}d{NUM_DATA}.png'
FIG_DPI: Final[int] = 600

# ================================

# The magnetic Ekman number
E_ETA: Final[float] = 0

CRITERION_C: Final[tuple[int, float]] = (N_C, R_C)

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

LIN_THETA: Final[np.ndarray] = np.linspace(0, math.pi, NUM_THETA)
LIN_MU: Final[np.ndarray] = np.cos(LIN_THETA)


def wrapper_plot_allfrobenius(
        bundle: tuple[np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray]) -> None:
    """A wrapper of a function to plot a figure of the discontinuity in
    the coefficient of the first Frobenius series solution for all the
    eigenfunctions

    Parameters
    ----------
    bundle : tuple of ndarray
        A tuple of results

    """

    fig: plt.Figure
    ax1: np.ndarray
    ax2: list
    ymin1: float
    ymax1: float
    ymin2: float
    ymax2: float
    (fig, ax1, ax2), [ymin1, ymax1], [ymin2, ymax2] \
        = plot_allfrobenius(bundle)

    ax1[0].grid()
    ax1[1].grid()
    ax1[0].set_axisbelow(True)
    ax1[1].set_axisbelow(True)

    ax1[0].set_xlim(-M_ORDER*ALPHA, M_ORDER*ALPHA)
    ax1[1].set_xlim(-M_ORDER*ALPHA, M_ORDER*ALPHA)

    ax1[0].set_ylim(ymin1, ymax1)
    ax1[1].set_ylim(ymin1, ymax1)
    ax2[0].set_ylim(ymin2, ymax2)
    ax2[1].set_ylim(ymin2, ymax2)

    ax1[0].set_xlabel(r'$\lambda=\omega/2\Omega_0$', fontsize=16)
    ax1[1].set_xlabel(r'$\lambda=\omega/2\Omega_0$', fontsize=16)

    ax1[0].set_ylabel(
        r'$\mathrm{sgn}(\mu_\mathrm{c})'
        + r'[C_\mathrm{I}^{(\mathrm{p})}-C_\mathrm{I}^{(\mathrm{e})}]$'
        + r', $C_\mathrm{I\!I}$', fontsize=16)
    ax1[1].set_ylabel(
        r'$\mathrm{sgn}(\mu_\mathrm{c})'
        + r'[C_\mathrm{I}^{(\mathrm{p})}-C_\mathrm{I}^{(\mathrm{e})}]$'
        + r', $C_\mathrm{I\!I}$', fontsize=16)
    ax2[0].set_ylabel(r'$I_\mathrm{num}$', fontsize=16)
    ax2[1].set_ylabel(r'$I_\mathrm{num}$', fontsize=16)

    ax1[0].set_title('Sinuous', fontsize=16)
    ax1[1].set_title('Varicose', fontsize=16)

    fig.suptitle(
        'Coefficients of the Frobenius solutions '
        + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
        + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}',
        fontsize=16)

    handle1, label1 = ax1[0].get_legend_handles_labels()
    handle2, label2 = ax2[0].get_legend_handles_labels()
    leg2_1: plt.Legend \
        = ax2[0].legend(handle1 + handle2, label1 + label2,
                        loc='best', fontsize=11)
    leg2_2: plt.Legend \
        = ax2[1].legend(handle1 + handle2, label1 + label2,
                        loc='best', fontsize=11)
    leg2_1.get_frame().set_alpha(1)
    leg2_2.get_frame().set_alpha(1)

    ax1[0].tick_params(labelsize=14)
    ax1[1].tick_params(labelsize=14)
    ax2[0].tick_params(labelsize=14)
    ax2[1].tick_params(labelsize=14)
    ax1[0].minorticks_on()
    ax1[1].minorticks_on()
    ax2[0].minorticks_on()
    ax2[1].minorticks_on()

    fig.tight_layout()

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / NAME_FIG
    fig.savefig(path_fig, dpi=FIG_DPI)
#


def plot_allfrobenius(
        bundle_with_vec: tuple[np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]) \
        -> tuple[tuple[plt.Figure, np.ndarray, list],
                 list[float], list[float]]:
    """Plots a figure of the discontinuity in the coefficient of the
    first Frobenius series solution for all the eigenfunctions

    Parameters
    ----------
    bundle_with_vec : tuple of ndarray
        A tuple of results

    Returns
    ----------
    fig_bundle : tuple
        A tuple of figures
    [ymin1, ymax1] : list of float
        The y limits of a graph
    [ymin2, ymax2] : list of float
        The y limits of a graph

    """

    psi_vec: np.ndarray
    vpa_vec: np.ndarray
    eig: np.ndarray
    sym: np.ndarray
    psi_vec, vpa_vec, eig, _, _, _, sym = bundle_with_vec

    fig: plt.Figure
    ax1: np.ndarray
    fig, ax1 = plt.subplots(1, 2, figsize=(10, 5))
    ax2: list = [ax1[0].twinx(), ax1[1].twinx()]

    c_1_jump: list[np.ndarray] = [np.zeros(SIZE_MAT), ] * 3
    c_2: list[np.ndarray] = [np.zeros(SIZE_MAT), ] * 3
    integral: list[np.ndarray] \
        = [np.zeros(SIZE_MAT, dtype=np.complex128), ] * 3

    for i_mode in range(SIZE_MAT):

        if math.isnan(eig[i_mode].real):
            continue
        #

        c_1_jump[0][i_mode], c_2[0][i_mode], integral[0][i_mode] \
            = calc_jump(
            psi_vec[:, i_mode], vpa_vec[:, i_mode], eig[i_mode])
    #

    sinuous: np.ndarray
    varicose: np.ndarray

    sinuous, varicose = sort_sv(sym)
    eig_s: np.ndarray = eig * sinuous
    eig_v: np.ndarray = eig * varicose

    c_1_jump[1] = c_1_jump[0] * sinuous
    c_1_jump[2] = c_1_jump[0] * varicose
    c_2[1] = c_2[0] * sinuous
    c_2[2] = c_2[0] * varicose
    integral[1] = integral[0] * sinuous
    integral[2] = integral[0] * varicose

    ax1[0].scatter(
        eig_s.real, c_1_jump[1], s=1, color='red',
        label=r'$\mathrm{sgn}(\mu_\mathrm{c})'
        + r'[C_\mathrm{I}^{(\mathrm{p})}-C_\mathrm{I}^{(\mathrm{e})}]$')
    ax1[1].scatter(
        eig_v.real, c_1_jump[2], s=1, color='red',
        label=r'$\mathrm{sgn}(\mu_\mathrm{c})'
        + r'[C_\mathrm{I}^{(\mathrm{p})}-C_\mathrm{I}^{(\mathrm{e})}]$')

    ax1[0].scatter(eig_s.real, c_2[1], s=1, color='blue',
                   label=r'$C_\mathrm{I\!I}$')
    ax1[1].scatter(eig_v.real, c_2[2], s=1, color='blue',
                   label=r'$C_\mathrm{I\!I}$')

    if np.nanmax(integral[0].imag) > 0:
        ax2[0].scatter(eig_s.real, integral[1].real, s=1, color='black',
                       label=r'$\mathrm{Re}(I_\mathrm{num})$')
        ax2[1].scatter(eig_v.real, integral[2].real, s=1, color='black',
                       label=r'$\mathrm{Re}(I_\mathrm{num})$')
        ax2[0].scatter(eig_s.real, integral[1].imag, s=1, color='grey',
                       label=r'$\mathrm{Im}(I_\mathrm{num})$')
        ax2[1].scatter(eig_v.real, integral[2].imag, s=1, color='grey',
                       label=r'$\mathrm{Im}(I_\mathrm{num})$')
    else:
        ax2[0].scatter(eig_s.real, integral[1].real, s=1, color='black',
                       label=r'$I_\mathrm{num}$')
        ax2[1].scatter(eig_v.real, integral[2].real, s=1, color='black',
                       label=r'$I_\mathrm{num}$')
    #

    i_almost_max: int = int(SIZE_MAT * 0.95)

    ymax1_candidate: list[float] = [
        sorted(np.abs(c_1_jump[0]))[i_almost_max],
        sorted(np.abs(c_2[0]))[i_almost_max]
    ]
    ymax1: float = max(ymax1_candidate[0], ymax1_candidate[1])
    ymin1: float = -ymax1

    ymax2: float = sorted(np.abs(integral[0]))[i_almost_max]
    ymin2: float = -ymax2

    fig_bundle: tuple[plt.Figure, np.ndarray, list] = (fig, ax1, ax2)

    return fig_bundle, [ymin1, ymax1], [ymin2, ymax2]
#


def calc_jump(psi_vec: np.ndarray,
              vpa_vec: np.ndarray,
              eig: complex) -> tuple[float, float, float]:
    """Calculates the discontinuity in the coefficient of the first
    Frobenius series solution

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
    c_1_jump : float
        The discontinuity in the coefficient of
            the first Frobenius series solution
    c_2 : float
        The coefficient of the second Frobenius series solution
    integral : float
        The integral in the expression of the discontinuity
            in the coefficient of the first Frobenius series solution

    """

    mu_c_tmp: complex = eig / (M_ORDER * ALPHA)
    mu_c: complex
    if 0 <= np.arccos(mu_c_tmp).real < math.pi/2:
        mu_c = mu_c_tmp
    else:
        mu_c = -mu_c_tmp
    #
    theta_c: float = np.arccos(mu_c).real
    i_theta_c: int = int(np.argmin(np.abs(LIN_THETA-theta_c)))

    c_1_jump: float
    c_2: float
    integral: float

    if (i_theta_c < NUM_DATA) or (NUM_THETA < i_theta_c):
        c_1_jump = math.nan
        c_2 = math.nan
        integral = math.nan

        return c_1_jump, c_2, integral
    #

    psi: np.ndarray = np.array([])
    psi1: np.ndarray = np.array([])
    psi2: np.ndarray = np.array([])

    psi, _ = make_eigf(psi_vec, vpa_vec, M_ORDER, PNM_NORM)
    psi1, psi2 = calc_frobenius(M_ORDER, ALPHA, NUM_THETA, eig, mu_c)

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

    y_eq = make_fitting_data(psi, psi2, NUM_DATA, i_theta_c, 'eq')
    x_eq = make_fitting_data(psi1, psi2, NUM_DATA, i_theta_c, 'eq')
    [a2_eq, b2_eq] = np.polyfit(x_eq, y_eq, 1)
    y_pole = make_fitting_data(psi, psi2, NUM_DATA, i_theta_c, 'pole')
    x_pole = make_fitting_data(psi1, psi2, NUM_DATA, i_theta_c, 'pole')
    [a2_pole, b2_pole] = np.polyfit(x_pole, y_pole, 1)

    c_1_eq: float = (b1_eq + a2_eq) / 2
    c_1_pole: float = (b1_pole + a2_pole) / 2

    c_1_jump = c_1_pole - c_1_eq
    c_2 = (a1_pole + a1_eq + b2_pole + b2_eq) / 4

    integral = -(c_1_jump/c_2) * (1-(mu_c**2)) \
        * (M_ORDER**2) * (ALPHA**2) * (2*mu_c) * (psi1[i_theta_c]**2)

    return c_1_jump, c_2, integral
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    caffeine.on(display=False)

    PNM_NORM: Final[np.ndarray] = load_legendre(M_ORDER, N_T, NUM_THETA)
    results_with_vec: tuple[np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray] \
        = wrapper_solve_eig(
        M_ORDER, ALPHA, E_ETA, SIZE_SUBMAT, CRITERION_C)

    plt.rcParams['text.usetex'] = True

    wrapper_plot_allfrobenius(results_with_vec)

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
