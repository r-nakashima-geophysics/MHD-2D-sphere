"""A code for 2D MHD waves on a rotating sphere under the non-Malkus
field B_phi = B_0 sin(theta) cos(theta)

Plots 2 figures (north-south 1D plot and 2D contour map) of the
eigenfunction of a chosen eigenmode.

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
from typing import Final

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from package_common.decorator_yesno import exe_yes_continue
from package_mhd2dsphere._make_eigf import (amp_range, choose_eigf, make_eigf,
                                            make_eigf_grid)
from package_mhd2dsphere._make_legendre import load_legendre
from package_mhd2dsphere.solve_eig import wrapper_solve_eig

# ========== Parameters ==========

# The boolean value to switch whether to display the value of the
# magnetic Ekman number or not when E_ETA = 0
SWITCH_DISP_ETA: Final[bool] = False

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = 0.1

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The truncation degree
N_T: Final[int] = 2000

# The number of the grid in the theta direction
NUM_THETA: Final[int] = 3601
NUM_THETA_SKIP: Final[int] = 181

# A criterion for convergence
# degree
N_C: Final[int] = int(N_T/2)
# ratio
R_C: Final[float] = 100

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_sincos_eigf'
NAME_FIG: Final[str] = 'MHD2Dsphere_sincos_eigf' \
    + f'_m{M_ORDER}a{ALPHA}E{E_ETA}N{N_T}th{NUM_THETA}'
NAME_FIG_SUFFIX: Final[tuple[str, str]] = ('_ns.png', '_map.png')
FIG_DPI: Final[int] = 600

# ================================

NUM_PHI: Final[int] = 361

CRITERION_C: Final[tuple[int, float]] = (N_C, R_C)

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

LIN_THETA: Final[np.ndarray] = np.linspace(0, math.pi, NUM_THETA)
LIN_THETA_SKIP: Final[np.ndarray] \
    = np.linspace(0, math.pi, NUM_THETA_SKIP)
LIN_PHI: Final[np.ndarray] = np.linspace(0, 2 * math.pi, NUM_PHI)

GRID_PHI: np.ndarray
GRID_THETA: np.ndarray
GRID_PHI, GRID_THETA \
    = np.meshgrid(LIN_PHI, LIN_THETA_SKIP[1:-1])

GRID_LAT: Final[np.ndarray] = np.rad2deg(
    np.full_like(GRID_THETA, math.pi/2) - GRID_THETA)
GRID_LON: Final[np.ndarray] = np.rad2deg(GRID_PHI)


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

    wrapper_plot_eigf(psi_vec, vpa_vec, eig, i_chosen)
#


def wrapper_plot_eigf(psi_vec: np.ndarray,
                      vpa_vec: np.ndarray,
                      eig: complex,
                      i_mode: int) -> None:
    """A wrapper of functions to plot figures of the eigenfunction of a
    chosen eigenmode

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

    psi_grid: np.ndarray
    vpa_grid: np.ndarray
    psi_grid, vpa_grid \
        = make_eigf_grid(psi_vec, vpa_vec, M_ORDER,
                         NUM_PHI, PNM_NORM_SKIP)

    plot_ns(psi, vpa, eig, i_mode)
    plot_map(psi_grid, vpa_grid, eig, i_mode)

    plt.show()
#


def plot_ns(psi: np.ndarray,
            vpa: np.ndarray,
            eig: complex,
            i_mode: int) -> None:
    """Plots a figure of the eigenfunction of a chosen eigenmode
    (north-south 1D plot)

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

    amp_max: float
    amp_min: float
    amp_max, amp_min = amp_range(psi, vpa)

    axis.grid()
    axis.set_xlim(0, math.pi)
    axis.set_xticks([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi])
    axis.set_xticklabels(['$0$', '$45$', '$90$', '$135$', '$180$'])
    axis.set_ylim(amp_min, amp_max)

    axis.set_xlabel('colatitude [degree]', fontsize=16)
    axis.set_ylabel('amplitude', fontsize=16)

    if eig.imag == 0:
        axis.set_title(
            r'$\lambda=$' + f' {eig.real:8.5f}',
            fontsize=16)
    else:
        axis.set_title(
            r'$\lambda=$' + f' {eig.real:8.5f} ' + r'$+$'
            + f'{eig.imag:8.5f} ' + r'$\mathrm{i}$',
            fontsize=16)

    if (not SWITCH_DISP_ETA) and (E_ETA == 0):
        fig.suptitle(
            r'Eigenfunction [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}',
            fontsize=16)
    else:
        fig.suptitle(
            r'Eigenfunction [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}, '
            + r'$E_\eta=$' + f' {E_ETA}', fontsize=16)

    leg: plt.Legend = axis.legend(loc='best', fontsize=11)
    leg.get_frame().set_alpha(1)

    axis.tick_params(labelsize=14)
    axis.minorticks_on()

    fig.tight_layout()

    name_fig_full: str = NAME_FIG + f'_{i_mode+1}' + NAME_FIG_SUFFIX[0]

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / name_fig_full
    fig.savefig(path_fig, dpi=FIG_DPI)
#


def plot_map(psi_grid: np.ndarray,
             vpa_grid: np.ndarray,
             eig: complex,
             i_mode: int) -> None:
    """Plots a figure of the eigenfunction of a chosen eigenmode (2D
    contour map)

    Parameters
    ----------
    psi_grid : ndarray
        A meshgrid of the stream function (psi)
    vpa_grid : ndarray
        A meshgrid of the vector potential (a)
    eig : complex
        An eigenvalue
    i_mode : int
        The index of a mode that you chose

    """

    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5),
        subplot_kw={'projection':
                    ccrs.Mollweide(central_longitude=0.0)})

    cfs: list = [None, ] * 2

    max_psi: float = np.nanmax(np.abs(psi_grid))
    max_vpa: float = np.nanmax(np.abs(vpa_grid))

    level_psi: np.ndarray \
        = np.arange(-max_psi, 1.2*max_psi, 0.2*max_psi)
    level_vpa: np.ndarray \
        = np.arange(-max_vpa, 1.2*max_vpa, 0.2*max_vpa)

    cfs[0] = axes[0].contourf(
        GRID_LON, GRID_LAT, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(),
        vmin=-max_psi, vmax=max_psi, cmap='bwr_r')
    axes[0].contour(
        GRID_LON, GRID_LAT, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(), colors='k',  linewidths=0.8)

    cfs[1] = axes[1].contourf(
        GRID_LON, GRID_LAT, vpa_grid, levels=level_vpa,
        transform=ccrs.PlateCarree(),
        vmin=-max_vpa, vmax=max_vpa, cmap='PiYG_r')
    axes[1].contour(
        GRID_LON, GRID_LAT, vpa_grid, levels=level_vpa,
        transform=ccrs.PlateCarree(), colors='k',  linewidths=0.8)

    axes[0].gridlines(linestyle=':')
    axes[1].gridlines(linestyle=':')

    cbar1 = fig.colorbar(cfs[0], ax=axes[0], orientation='horizontal')
    cbar2 = fig.colorbar(cfs[1], ax=axes[1], orientation='horizontal')
    cbar1.ax.tick_params(labelsize=14)
    cbar2.ax.tick_params(labelsize=14)

    axes[0].set_title(r'stream function $\psi_1$', fontsize=16)
    axes[1].set_title(
        r'vector potential $\mathrm{sgn}(\alpha)a_1'
        + r'/\sqrt{\rho_0\mu_\mathrm{m}}$', fontsize=16)

    if (not SWITCH_DISP_ETA) and (E_ETA == 0):
        fig.suptitle(
            r'Eigenfunction [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}\n\n'
            + r'$\lambda=$' + f' {eig.real:8.5f}',
            fontsize=16)
    elif eig.imag == 0:
        fig.suptitle(
            r'Eigenfunction [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}, '
            + r'$E_\eta=$' + f' {E_ETA}\n\n'
            + r'$\lambda=$' + f' {eig.real:8.5f}',
            fontsize=16)
    else:
        fig.suptitle(
            r'Eigenfunction [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}, '
            + r'$E_\eta=$' + f' {E_ETA}\n\n'
            + r'$\lambda=$' + f' {eig.real:8.5f} ' + r'$+$'
            + f'{eig.imag:8.5f} ' + r'$\mathrm{i}$',
            fontsize=16)

    fig.tight_layout()

    name_fig_full: str = NAME_FIG + f'_{i_mode+1}' + NAME_FIG_SUFFIX[1]

    path_fig: Path = PATH_DIR_FIG / name_fig_full
    fig.savefig(path_fig, dpi=FIG_DPI)
#


if __name__ == '__main__':

    PNM_NORM: Final[np.ndarray] \
        = load_legendre(M_ORDER, N_T, NUM_THETA)
    PNM_NORM_SKIP: Final[np.ndarray] \
        = load_legendre(M_ORDER, N_T, NUM_THETA_SKIP)
    results: tuple[np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray] \
        = wrapper_solve_eig(
        M_ORDER, ALPHA, E_ETA, SIZE_SUBMAT, CRITERION_C)

    plt.rcParams['text.usetex'] = True

    wrapper_choose_eigf(results)
#
