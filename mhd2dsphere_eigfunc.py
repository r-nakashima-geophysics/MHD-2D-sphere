"""A Python script to plot figures of the eigenfunction of a chosen eigenmode
for two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and a background zonal flow, U_phi = U_0 U(theta) sin(theta).

This script can create two figures: a north-south 1D plot and a 2D contour map
of the eigenfunction of a chosen eigenmode.

Notes
----------
All other parameters aside from command line arguments are described within the
script.

References
----------
[1] Ryosuke Nakashima, Shigeo Yoshida, Two-dimensional ideal
magnetohydrodynamic waves on a rotating sphere under a non-Malkus field: I.
Continuous spectrum and its ray-theoretical interpretation. Geophysical &
Astrophysical Fluid Dynamics 118(5-6), 387-440 (2024). doi:
10.1080/03091929.2024.2384388

[2] Ryosuke Nakashima, Shigeo Yoshida (in prep.)

Examples
--------
$ python3 mhd2dsphere_eigfunc.py
"""

from pathlib import Path
from typing import Final

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from package_common.background_field import BackgroundField
from package_common.common_types import ArrayComplex, ArrayFloat
from package_common.decorator_yesno import exe_yes_continue
from package_common.default_logger import DefaultLogger
from package_common.default_plotter import (Colorbar, DefaultPlotter,
                                            QuadContourSet, create_plotter)
from package_common.spectral_deform import (ComplexCoordinate,
                                            init_complex_coordinate_simple)
from package_mhd2dsphere import init_background_b, init_background_u
from package_mhd2dsphere.make_eigfunc import (amp_range, choose_eigfunc,
                                              create_basis, make_eigfunc,
                                              make_eigfunc_grid)
from package_mhd2dsphere.solve_eig import (prepare_chebyshev_gauss_quad,
                                           wrapper_solve_eig)
from package_mhd2dsphere.typed_dict import (DictBackgroundField,
                                            DictChebyshevGaussQuad,
                                            DictCriterionC, DictEigenmodeInfo,
                                            DictResult)

# ========== Parameters ==========

# Background field
BG_FIELD_B: Final[BackgroundField] = init_background_b.b_malkus('mu')
BG_FIELD_U: Final[BackgroundField] = init_background_u.u_rigid('mu')
# For the spectral deformation method
MU_COMPLEX: Final[ComplexCoordinate] = init_complex_coordinate_simple(
    -1, 1, alpha=0, beta_0=0, beta_1=0)
# The boolean value to switch whether to follow Nakashima & Yoshida
# (2024)[1]_ or not
# If SWITCH_NY24 is True, BG_FIELD_B, BG_FIELD_U and
# MU_COMPLEX are ignored.
SWITCH_NY24: Final[bool] = False

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = 0.1

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The Rossby number
ROSSBY: Final[float] = 0

# The truncation degree
N_T: Final[int] = 500
# N_T: Final[int] = 2000

# The number of grid points in the theta and phi directions
NUM_THETA: Final[int] = 3601
NUM_THETA_SKIP: Final[int] = 181
NUM_PHI: Final[int] = 361

# The criterion for convergence
# degree
N_C: Final[int] = int(N_T/2)
# ratio
R_C: Final[float] = 100

# The paths and filenames of outputs
PATH_DIR: Final[Path] = Path('.') / 'fig' / 'MHD2Dsphere_eigfunc'
NAME_FIG: Final[str] \
    = f'MHD2Dsphere_eigfunc_NY24_m={M_ORDER}_E={E_ETA}_N={N_T}' \
    if SWITCH_NY24 \
    else f'MHD2Dsphere_eigfunc_B{BG_FIELD_B.name}U{BG_FIELD_U.name}' \
    + f'_m={M_ORDER}_E={E_ETA}_R={ROSSBY}_N={N_T}' \
    + f'_{MU_COMPLEX.name}'
NAME_FIG_SUFFIX: Final[tuple[str, str]] = ('_1d.png', '_2d.png')
FIG_DPI: Final[int] = 600

# The boolean value to switch whether to display the value of the
# magnetic Ekman number or not when E_ETA = 0
SWITCH_DISP_ETA: Final[bool] = False

# ================================

BG_FIELD: Final[DictBackgroundField] = {
    'B': BG_FIELD_B,
    'U': BG_FIELD_U,
    'MU': MU_COMPLEX,
    'NY24': SWITCH_NY24,
}

TEX_BG_FIELD: Final[str] \
    = r'$B_{0\phi}=B_0\sin\theta\cos\theta$, $U_{0\phi}=0$' \
    if SWITCH_NY24 \
    else f'{BG_FIELD_B.tex}, {BG_FIELD_U.tex}'

CRITERION_C: Final[DictCriterionC] = {
    'degree': N_C,
    'ratio': R_C
}

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1 if SWITCH_NY24 else N_T + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

LIN_THETA: Final[ArrayFloat] = np.linspace(0, np.pi, NUM_THETA)
LIN_THETA_SKIP: Final[ArrayFloat] = np.linspace(0, np.pi, NUM_THETA_SKIP)
LIN_PHI: Final[ArrayFloat] = np.linspace(0, 2*np.pi, NUM_PHI)

GRID_PHI: np.ndarray
GRID_THETA: np.ndarray
GRID_PHI, GRID_THETA = np.meshgrid(LIN_PHI, LIN_THETA_SKIP[1:-1])

GRID_LAT: Final[ArrayFloat] = np.rad2deg(
    np.full_like(GRID_THETA, np.pi/2) - GRID_THETA)
GRID_LON: Final[ArrayFloat] = np.rad2deg(GRID_PHI)

TEXT_TITLE: Final[str] \
    = f'Eigenfunction [{TEX_BG_FIELD}] : ' \
    + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}, ' \
    + r'$R=$' + f' {ROSSBY}' \
    if ((not SWITCH_DISP_ETA) and (E_ETA == 0)) \
    else f'Eigenfunction [{TEX_BG_FIELD}] : ' \
    + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}, ' \
    + r'$E_\eta=$' + f' {E_ETA}, ' + r'$R=$' + f' {ROSSBY}'


@exe_yes_continue
def wrapper_choose_eigfunc(results: DictResult,
                           *,
                           basis_func: ArrayFloat | ArrayComplex,
                           basis_func_skip: ArrayFloat | ArrayComplex) -> None:
    """Choose eigenmodes which you want to plot.

    Parameters
    ----------
    results : DictResult
        A dictionary of results of the eigenvalue problem.
    basis_func : ArrayFloat | ArrayComplex
        The values of basis functions at grid points.
    basis_func_skip : ArrayFloat | ArrayComplex
        The values of basis functions at grid points.
    """

    result: DictEigenmodeInfo = choose_eigfunc(results, SIZE_MAT)

    wrapper_plot_eigfunc(
        result, basis_func=basis_func, basis_func_skip=basis_func_skip)


def wrapper_plot_eigfunc(result: DictEigenmodeInfo,
                         *,
                         basis_func: ArrayFloat | ArrayComplex,
                         basis_func_skip: ArrayFloat | ArrayComplex) -> None:
    """Plot figures of the eigenfunction of a chosen eigenmode.

    Parameters
    ----------
    result : DictEigenmodeInfo
        A dictionary of result of an eigenmode which you chose.
    basis_func : ArrayFloat | ArrayComplex
        The values of basis functions at grid points.
    basis_func_skip : ArrayFloat | ArrayComplex
        The values of basis functions at grid points.
    """

    eig: complex = result['eig']
    i_mode: int = result['i_mode']

    psi: ArrayComplex
    vpa: ArrayComplex
    psi, vpa = make_eigfunc(result, M_ORDER, LIN_THETA, basis_func,
                            background_field=BG_FIELD)

    psi_grid: ArrayFloat
    vpa_grid: ArrayFloat
    psi_grid, vpa_grid = make_eigfunc_grid(
        result, M_ORDER, LIN_THETA_SKIP, LIN_PHI, basis_func_skip,
        background_field=BG_FIELD)

    plot_ns(psi, vpa, eig, i_mode)
    plot_map(psi_grid, vpa_grid, eig, i_mode)

    plt.show()


def plot_ns(psi: np.ndarray,
            vpa: np.ndarray,
            eig: complex,
            i_mode: int) -> None:
    """Plot a figure of the eigenfunction of a chosen eigenmode
    (north-south 1D plot).

    Parameters
    ----------
    psi : ndarray
        The stream function (psi) of an eigenmode which you chose.
    vpa : ndarray
        The vector potential (a) of an eigenmode which you chose.
    eig : complex
        The eigenvalue of an eigenmode which you chose.
    i_mode : int
        The index of an eigenmode which you chose.
    """

    plotter: DefaultPlotter = create_plotter(1, 1, figsize=(7, 4))

    plotter.axes.plot(LIN_THETA, psi.real, color='red',
                      label=r'stream function $\tilde{\psi}$')
    plotter.axes.plot(LIN_THETA, vpa.real, color='blue',
                      label=r'vector potential $\mathrm{sgn}(\alpha)\tilde{a}/'
                      + r'\sqrt{\rho_0\mu_\mathrm{m}}$')
    if np.nanmax(np.abs(psi.imag)) > 0:
        plotter.axes.plot(LIN_THETA, psi.imag, color='red', linestyle=':')

    if np.nanmax(np.abs(vpa.imag)) > 0:
        plotter.axes.plot(LIN_THETA, vpa.imag, color='blue', linestyle=':')

    amp_max: float
    amp_min: float
    amp_max, amp_min = amp_range(psi, vpa)

    plotter.axes.set_xlim(0, np.pi)
    plotter.axes.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    plotter.axes.set_xticklabels(['$0$', '$45$', '$90$', '$135$', '$180$'])
    plotter.axes.set_ylim(amp_min, amp_max)

    plotter.axes.set_xlabel('colatitude [degree]', fontsize=16)
    plotter.axes.set_ylabel('amplitude', fontsize=16)

    if eig.imag == 0:
        plotter.axes.set_title(
            r'$\lambda=$' + f' {eig.real:8.5f}', fontsize=16)
    else:
        plotter.axes.set_title(
            r'$\lambda=$' + f' {eig.real:8.5f} ' + r'$+$'
            + f'{eig.imag:8.5f} ' + r'$\mathrm{i}$', fontsize=16)

    plotter.fig.suptitle(TEXT_TITLE, fontsize=16)

    plotter.leg = plotter.axes.legend(loc='best', fontsize=11)

    plotter.axes.tick_params(labelsize=14)

    name_fig: str = NAME_FIG + f'_{i_mode+1}' + NAME_FIG_SUFFIX[0]
    plotter.save(PATH_DIR, name_fig, FIG_DPI)


def plot_map(psi_grid: np.ndarray,
             vpa_grid: np.ndarray,
             eig: complex,
             i_mode: int) -> None:
    """Plot a figure of the eigenfunction of a chosen eigenmode (2D
    contour map).

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

    plotter: DefaultPlotter = create_plotter(
        1, 2, figsize=(10, 5),
        subplot_kw={'projection':
                    ccrs.Mollweide(central_longitude=0.0)})

    max_psi: float = np.nanmax(np.abs(psi_grid))
    max_vpa: float = np.nanmax(np.abs(vpa_grid))

    level_psi: np.ndarray \
        = np.arange(-max_psi, 1.2*max_psi, 0.2*max_psi)
    level_vpa: np.ndarray \
        = np.arange(-max_vpa, 1.2*max_vpa, 0.2*max_vpa)

    contour1: QuadContourSet = plotter.axes[0].contourf(
        GRID_LON, GRID_LAT, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(),
        vmin=-max_psi, vmax=max_psi, cmap='bwr_r')
    plotter.axes[0].contour(
        GRID_LON, GRID_LAT, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(), colors='k',  linewidths=0.8)

    contour2: QuadContourSet = plotter.axes[1].contourf(
        GRID_LON, GRID_LAT, vpa_grid, levels=level_vpa,
        transform=ccrs.PlateCarree(),
        vmin=-max_vpa, vmax=max_vpa, cmap='PiYG_r')
    plotter.axes[1].contour(
        GRID_LON, GRID_LAT, vpa_grid, levels=level_vpa,
        transform=ccrs.PlateCarree(), colors='k',  linewidths=0.8)

    plotter.axes[0].gridlines(linestyle=':')
    plotter.axes[1].gridlines(linestyle=':')

    cbar1: Colorbar = plotter.fig.colorbar(
        contour1, ax=plotter.axes[0], orientation='horizontal')
    cbar2: Colorbar = plotter.fig.colorbar(
        contour2, ax=plotter.axes[1], orientation='horizontal')
    cbar1.ax.tick_params(labelsize=14)
    cbar2.ax.tick_params(labelsize=14)

    plotter.axes[0].set_title(r'stream function $\psi_1$', fontsize=16)
    plotter.axes[1].set_title(
        r'vector potential $\mathrm{sgn}(\alpha)a_1'
        + r'/\sqrt{\rho_0\mu_\mathrm{m}}$', fontsize=16)

    if np.isclose(eig.imag, 0):
        plotter.fig.suptitle(
            TEXT_TITLE + '\n\n' + r'$\lambda=$' + f' {eig.real:8.5f}',
            fontsize=16)
    else:
        plotter.fig.suptitle(
            TEXT_TITLE + '\n\n'
            + r'$\lambda=$' + f' {eig.real:8.5f} ' + r'$+$'
            + f'{eig.imag:8.5f} ' + r'$\mathrm{i}$',
            fontsize=16)

    name_fig: str = NAME_FIG + f'_{i_mode+1}' + NAME_FIG_SUFFIX[1]
    plotter.save(PATH_DIR, name_fig, FIG_DPI)


if __name__ == '__main__':

    logger: DefaultLogger = DefaultLogger(__name__)

    if SWITCH_NY24:
        logger.show_params(f'{SWITCH_NY24=}',
                           f'{M_ORDER=}',
                           f'{ALPHA=}',
                           f'{E_ETA=}',
                           f'{ROSSBY=}',
                           f'{N_T=}')
    else:
        logger.show_params(f'{BG_FIELD_B.name=}',
                           f'{BG_FIELD_U.name=}',
                           f'{MU_COMPLEX.name=}',
                           f'{M_ORDER=}',
                           f'{ALPHA=}',
                           f'{E_ETA=}',
                           f'{ROSSBY=}',
                           f'{N_T=}')

    basis: ArrayFloat | ArrayComplex = create_basis(
        M_ORDER, N_T, LIN_THETA, background_field=BG_FIELD)
    basis_skip: ArrayFloat | ArrayComplex = create_basis(
        M_ORDER, N_T, LIN_THETA_SKIP, background_field=BG_FIELD)

    quad: DictChebyshevGaussQuad | None = prepare_chebyshev_gauss_quad(
        M_ORDER, ROSSBY, SIZE_SUBMAT, background_field=BG_FIELD)

    data: DictResult = wrapper_solve_eig(
        M_ORDER, ALPHA, E_ETA, ROSSBY, SIZE_SUBMAT,
        criterion_c=CRITERION_C, background_field=BG_FIELD, dict_quad=quad)

    wrapper_choose_eigfunc(data, basis_func=basis, basis_func_skip=basis_skip)
