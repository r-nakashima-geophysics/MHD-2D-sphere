"""A code for 2D MHD waves on a rotating sphere under the non-Malkus
field B_phi = B_0 sin(theta) cos(theta)

Plots 1-2 figures displaying all the eigenfunctions for given
parameters.

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
    python3 mhd2dsphere_sincos_alleigf.py
In the below example, ALPHA will be set to 1.
    python3 mhd2dsphere_sincos_alleigf.py 1
"""

import math
import os
from pathlib import Path
from time import perf_counter
from typing import Final

import caffeine
import matplotlib.pyplot as plt
import numpy as np

from package_common.input_helper import input_value
from package_mhd2dsphere.load_data import load_legendre
from package_mhd2dsphere.make_eigf import make_eigf
from package_mhd2dsphere.processing_results import screening_eig_q
from package_mhd2dsphere.solve_eig import wrapper_solve_eig

# ========== Parameters ==========

# The boolean value to switch whether to display the value of the
# magnetic Ekman number when E_ETA = 0
SWITCH_DISP_ETA: Final[bool] = False

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = input_value(0.1, float)

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The truncation degree
N_T: Final[int] = 2000

# The number of the grid in the theta direction
NUM_THETA: Final[int] = 3601

# A criterion for convergence
# degree
N_C: Final[int] = int(N_T/2)
# ratio
R_C: Final[float] = 100

# A criterion for plotting eigenfunctions based on the Q value
CRITERION_Q: Final[float] = 0

# The range of eigenvalues (real part)
EIG_RE_INIT: Final[float] = -M_ORDER * ALPHA
EIG_RE_END: Final[float] = M_ORDER * ALPHA

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_sincos_alleigf'
NAME_FIG: Final[str] = 'MHD2Dsphere_sincos_alleigf' \
    + f'_m{M_ORDER}a{ALPHA}E{E_ETA}N{N_T}th{NUM_THETA}'
NAME_FIG_SUFFIX_1: Final[str] = f'q{CRITERION_Q}'
NAME_FIG_SUFFIX_2: Final[tuple[str, str]] = ('_R.png', '_I.png')
FIG_DPI: Final[int] = 600

# ================================

CRITERION_C: Final[tuple[int, float]] = (N_C, R_C)

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

LIN_THETA: Final[np.ndarray] = np.linspace(0, math.pi, NUM_THETA)


def wrapper_plot_alleigf(
        bundle: tuple[np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray]) -> None:
    """A wrapper of a function to plot a figure displaying all the
    eigenfunctions

    Parameters
    ----------
    bundle : tuple of ndarray
        A tuple of results

    """

    fig1: plt.Figure
    fig2: plt.Figure
    ax1: np.ndarray
    ax2: np.ndarray
    save_fig2: bool
    (fig1, fig2, ax1, ax2), save_fig2 = plot_alleigf(bundle)

    ax_all: tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes,
                  plt.Axes, plt.Axes, plt.Axes, plt.Axes,] \
        = (ax1[0, 0], ax1[0, 1], ax1[1, 0], ax1[1, 1],
           ax2[0, 0], ax2[0, 1], ax2[1, 0], ax2[1, 1])

    for axis in ax_all:
        axis.grid()
        axis.set_axisbelow(True)

        axis.set_xlim(EIG_RE_INIT, EIG_RE_END)
        axis.set_ylim(math.pi, 0)

        axis.set_yticks([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi])
        axis.set_yticklabels(['$0$', '$45$', '$90$', '$135$', '$180$'])

        axis.tick_params(labelsize=14)
        axis.minorticks_on()

    if save_fig2:
        ax1[1, 0].set_xlabel(
            r'$\mathrm{Re}(\lambda)=\mathrm{Re}(\omega)/2\Omega_0$',
            fontsize=16)
        ax1[1, 1].set_xlabel(
            r'$\mathrm{Re}(\lambda)=\mathrm{Re}(\omega)/2\Omega_0$',
            fontsize=16)
        ax2[1, 0].set_xlabel(
            r'$\mathrm{Re}(\lambda)=\mathrm{Re}(\omega)/2\Omega_0$',
            fontsize=16)
        ax2[1, 1].set_xlabel(
            r'$\mathrm{Re}(\lambda)=\mathrm{Re}(\omega)/2\Omega_0$',
            fontsize=16)
    else:
        ax1[1, 0].set_xlabel(
            r'$\lambda=\omega/2\Omega_0$', fontsize=16)
        ax1[1, 1].set_xlabel(
            r'$\lambda=\omega/2\Omega_0$', fontsize=16)

    ax1[0, 0].set_ylabel('colatitude [degree]', fontsize=16)
    ax1[1, 0].set_ylabel('colatitude [degree]', fontsize=16)
    ax2[0, 0].set_ylabel('colatitude [degree]', fontsize=16)
    ax2[1, 0].set_ylabel('colatitude [degree]', fontsize=16)

    if save_fig2:
        ax1[0, 0].set_title(r'Sinuous $|\mathrm{Re}(\tilde{\psi})|$',
                            fontsize=16)
        ax1[0, 1].set_title(r'Sinuous $|\mathrm{Re}(\tilde{a})|$',
                            fontsize=16)
        ax1[1, 0].set_title(r'Varicose $|\mathrm{Re}(\tilde{\psi})|$',
                            fontsize=16)
        ax1[1, 1].set_title(r'Varicose $|\mathrm{Re}(\tilde{a})|$',
                            fontsize=16)
        ax2[0, 0].set_title(r'Sinuous $|\mathrm{Im}(\tilde{\psi})|$',
                            fontsize=16)
        ax2[0, 1].set_title(r'Sinuous $|\mathrm{Im}(\tilde{a})|$',
                            fontsize=16)
        ax2[1, 0].set_title(r'Varicose $|\mathrm{Im}(\tilde{\psi})|$',
                            fontsize=16)
        ax2[1, 1].set_title(r'Varicose $|\mathrm{Im}(\tilde{a})|$',
                            fontsize=16)
    else:
        ax1[0, 0].set_title(r'Sinuous $|\tilde{\psi}|$',
                            fontsize=16)
        ax1[0, 1].set_title(r'Sinuous $|\tilde{a}|$',
                            fontsize=16)
        ax1[1, 0].set_title(r'Varicose $|\tilde{\psi}|$',
                            fontsize=16)
        ax1[1, 1].set_title(r'Varicose $|\tilde{a}|$',
                            fontsize=16)

    if (not SWITCH_DISP_ETA) and (E_ETA == 0):
        fig1.suptitle(
            r'Eigenfunctions [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}',
            fontsize=16)
        fig2.suptitle(
            r'Eigenfunctions [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}',
            fontsize=16)
    else:
        fig1.suptitle(
            r'Eigenfunctions [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}, '
            + r'$E_\eta=$' + f' {E_ETA}', fontsize=16)
        fig2.suptitle(
            r'Eigenfunctions [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}, '
            + r'$E_\eta=$' + f' {E_ETA}', fontsize=16)

    fig1.tight_layout()
    fig2.tight_layout()

    name_fig_full_1: str
    name_fig_full_2: str
    if (E_ETA != 0) and (CRITERION_Q > 0):
        name_fig_full_1 = NAME_FIG + NAME_FIG_SUFFIX_1
        name_fig_full_2 = NAME_FIG + NAME_FIG_SUFFIX_1
    else:
        name_fig_full_1 = NAME_FIG
        name_fig_full_2 = NAME_FIG

    name_fig_full_1 += NAME_FIG_SUFFIX_2[0]
    name_fig_full_2 += NAME_FIG_SUFFIX_2[1]

    os.makedirs(PATH_DIR_FIG, exist_ok=True)

    path_fig_1: Path = PATH_DIR_FIG / name_fig_full_1
    path_fig_2: Path = PATH_DIR_FIG / name_fig_full_2

    fig1.savefig(path_fig_1, dpi=FIG_DPI)
    if save_fig2:
        fig2.savefig(path_fig_2, dpi=FIG_DPI)

#


def plot_alleigf(bundle: tuple[np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]) \
        -> tuple[tuple[plt.Figure, plt.Figure, np.ndarray, np.ndarray],
                 bool]:
    """Plots a figure displaying all the eigenfunctions

    Parameters
    ----------
    bundle : tuple of ndarray
        A tuple of results

    Returns
    ----------
    fig_bundle : tuple
        A tuple of figures
    save_fig2 : bool
        The boolean value to determine whether to save a figure

    """

    if (E_ETA != 0) and (CRITERION_Q > 0):
        bundle = wrapper_screening_eig_q(bundle)

    psi_vec: np.ndarray
    vpa_vec: np.ndarray
    eig: np.ndarray
    sym: np.ndarray
    psi_vec, vpa_vec, eig, _, _, _, sym = bundle

    psi_all: np.ndarray
    vpa_all: np.ndarray
    psi_all, vpa_all = make_alleigf(eig, psi_vec, vpa_vec)

    cmap_max: float = cmap_range(psi_all, vpa_all)
    cmap_min: float = 0

    fig1: plt.Figure
    fig2: plt.Figure
    ax1: np.ndarray
    ax2: np.ndarray
 real part
    fig1, ax1 = plt.subplots(2, 2, figsize=(10, 10))
 imaginary part
    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10))

    plot_fig2: bool
    save_fig2: bool = False

    psi: list[np.ndarray] = [np.array([]), ] * 2
    vpa: list[np.ndarray] = [np.array([]), ] * 2

    for i_mode in range(SIZE_MAT):

        if math.isnan(eig[i_mode].real):
            continue
    

        psi[0] = np.abs(psi_all[i_mode, :].real)
        vpa[0] = np.abs(vpa_all[i_mode, :].real)
        psi[1] = np.abs(psi_all[i_mode, :].imag)
        vpa[1] = np.abs(vpa_all[i_mode, :].imag)

        ones_lambda: np.ndarray = np.full(NUM_THETA, eig[i_mode].real)

        plot_fig2 = False
        if (np.nanmax(psi[1]) > 0) or (np.nanmax(vpa[1]) > 0):
            plot_fig2 = True
            save_fig2 = True
    

        if sym[i_mode] == 'sinuous':
            ax1[0, 0].scatter(
                ones_lambda, LIN_THETA, s=0.001, c=psi[0],
                cmap='Purples', vmin=cmap_min, vmax=cmap_max)
            ax1[0, 1].scatter(
                ones_lambda, LIN_THETA, s=0.001, c=vpa[0],
                cmap='Purples', vmin=cmap_min, vmax=cmap_max)
            if plot_fig2:
                ax2[0, 0].scatter(
                    ones_lambda, LIN_THETA, s=0.001, c=psi[1],
                    cmap='Purples', vmin=cmap_min, vmax=cmap_max)
                ax2[0, 1].scatter(
                    ones_lambda, LIN_THETA, s=0.001, c=vpa[1],
                    cmap='Purples', vmin=cmap_min, vmax=cmap_max)
        
        elif sym[i_mode] == 'varicose':
            ax1[1, 0].scatter(
                ones_lambda, LIN_THETA, s=0.001, c=psi[0],
                cmap='Purples', vmin=cmap_min, vmax=cmap_max)
            ax1[1, 1].scatter(
                ones_lambda, LIN_THETA, s=0.001, c=vpa[0],
                cmap='Purples', vmin=cmap_min, vmax=cmap_max)
            if plot_fig2:
                ax2[1, 0].scatter(
                    ones_lambda, LIN_THETA, s=0.001, c=psi[1],
                    cmap='Purples', vmin=cmap_min, vmax=cmap_max)
                ax2[1, 1].scatter(
                    ones_lambda, LIN_THETA, s=0.001, c=vpa[1],
                    cmap='Purples', vmin=cmap_min, vmax=cmap_max)
        
    


    fig_bundle: tuple[plt.Figure, plt.Figure, np.ndarray, np.ndarray] \
        = (fig1, fig2, ax1, ax2)

    return fig_bundle, save_fig2
#


def wrapper_screening_eig_q(
        bundle: tuple[np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray]) \
    -> tuple[np.ndarray, np.ndarray,
             np.ndarray, np.ndarray, np.ndarray,
             np.ndarray, np.ndarray]:
    """A wrapper of a function to check the Q-values of eigenmodes

    Parameters
    ----------
    bundle : tuple of ndarray
        A tuple of results

    Returns
    ----------
    bundle : tuple of ndarray
        A tuple of results

    """

    psi_vec: np.ndarray
    vpa_vec: np.ndarray
    eig: np.ndarray
    mke: np.ndarray
    mme: np.ndarray
    ohm: np.ndarray
    sym: np.ndarray
    psi_vec, vpa_vec, eig, mke, mme, ohm, sym = bundle
    subbundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                     np.ndarray, np.ndarray, np.ndarray] \
        = (np.array(ALPHA), eig, mke, mme, ohm, sym)

    _, eig, mke, mme, ohm, sym = screening_eig_q(CRITERION_Q, subbundle)

    bundle = (psi_vec, vpa_vec, eig, mke, mme, ohm, sym)

    return bundle
#


def make_alleigf(eig: np.ndarray,
                 psi_vec: np.ndarray,
                 vpa_vec: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    """Makes all the eigenfunctions from an eigenvector

    Parameters
    ----------
    eig : ndarray
        Eigenvalues
    psi_vec : ndarray
        Eigenvectors of the stream function (psi)
    vpa_vec : ndarray
        Eigenvectors of the vector potential (a)

    Returns
    ----------
    psi_all : ndarray
        All the eigenfunctions of the stream function (psi)
    vpa_all : ndarray
        All the eigenfunctions of the vector potential (a)

    """

    psi_all: np.ndarray = \
        np.zeros((SIZE_MAT, NUM_THETA), dtype=np.complex128)
    vpa_all: np.ndarray = \
        np.zeros((SIZE_MAT, NUM_THETA), dtype=np.complex128)

    for i_mode in range(SIZE_MAT):

        if math.isnan(eig[i_mode].real):
            continue
    

        psi_all[i_mode, :], vpa_all[i_mode, :] = make_eigf(
            psi_vec[:, i_mode], vpa_vec[:, i_mode], M_ORDER, PNM_NORM)


    return psi_all, vpa_all
#


def cmap_range(psi_all: np.ndarray,
               vpa_all: np.ndarray) -> float:
    """Determines the range of the color map

    Parameters
    ----------
    psi_all : ndarray
        All the eigenfunctions of the stream function (psi)
    vpa_all : ndarray
        All the eigenfunctions of the vector potential (a)

    Returns
    ----------
    cmap_max : float
        The maximum value of the color map

    """

    i_almost_max: int = int(SIZE_MAT * NUM_THETA * 0.95)

    cmap_psi_max: float \
        = sorted(np.abs(np.ravel(psi_all)))[i_almost_max]
    cmap_vpa_max: float \
        = sorted(np.abs(np.ravel(vpa_all)))[i_almost_max]
    cmap_max: float = max(cmap_psi_max, cmap_vpa_max)

    return cmap_max
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    caffeine.on(display=False)

    PNM_NORM: Final[np.ndarray] = load_legendre(M_ORDER, N_T, NUM_THETA)
    results: tuple[np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray] \
        = wrapper_solve_eig(
        M_ORDER, ALPHA, E_ETA, SIZE_SUBMAT, CRITERION_C)

    plt.rcParams['text.usetex'] = True

    wrapper_plot_alleigf(results)

    TIME_ELAPSED: float = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
