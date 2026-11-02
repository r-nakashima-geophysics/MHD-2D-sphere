"""A Python script to plot the results of time-dependent (quasi-)linear
simulations for two-dimensional (2D) incompressible magnetohydrodynamic (MHD)
waves on a rotating sphere under a toroidal background field, B_phi = B_0
B(theta) sin(theta), and a background zonal flow, U_phi = U_0 U(theta)
sin(theta).

This script creates three figures: a 2D contour map, a time series plot, and
Hovmoller diagram.

Parameters
----------
ID : str
    The run index.

Notes
-----
All other parameters aside from command line arguments are described within the
script. Before executing this code, mhd2dsphere_linsim.py with the same
parameters must be run.

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
Run the script with the default value of `ID`.
    $ uv run python mhd2dsphere_linsim.py
Run the script with a specified value of `ID` (say 0001):
    $ uv run python mhd2dsphere_linsim.py 0001
"""

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from package_common.common_types import (ArrayComplex, ArrayFloat, ArrayInt,
                                         Final, cast)
from package_common.default_logger import DefaultLogger
from package_common.default_plotter import (Colorbar, DefaultGridPlotter,
                                            QuadContourSet, create_plotter)
from package_common.default_timer import DefaultTimer
from package_common.progress_bar import ProgressBar
from package_common.utils_collocation import create_cheb_invexpan_mat
from package_common.utils_input import input_string
from package_common.utils_name import create_function_name_progress_bar
from package_mhd2dsphere.load_data import load_linsim_results
from package_mhd2dsphere.typed_dict import DictLinsimResults

# ========== Parameters ========== #

# the run index
ID: Final[str] = input_string('0000')

# The Lehnert number
ALPHA: Final[float] = 0.1

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The Rossby number
ROSSBY: Final[float] = 0

# The truncation degree
N_T: Final[int] = 500

# The number of grid points in the phi direction
NUM_PHI: Final[int] = 361

# The maximum values for the color scales
MAX_PSI: Final[float] = 5
MAX_MVP: Final[float] = 5

# The path and filename of inputs
PATH_DIR_INPUT: Final[Path] = Path('.') / 'output' / 'MHD2Dsphere_linsim'
NAME_FILE: Final[str] = (
    f'MHD2Dsphere_linsim'
    + f'_a={ALPHA}_E={E_ETA}_R={ROSSBY}_N={N_T}_#{ID}.npz'
)

# The paths and filenames of outputs
NAME_DIR: Final[str] = (
    f'MHD2Dsphere_linsim'
    + f'_a={ALPHA}_E={E_ETA}_R={ROSSBY}_N={N_T}_#{ID}'
)
PATH_DIR_FIG: Final[Path] = Path('.') / 'fig' / 'MHD2Dsphere_linsim' / NAME_DIR
NAME_FIG: Final[tuple[str, str, str]] = ('2d', 'time', 'hovmoller')
NAME_FIG_SUFFIX: Final[str] = '.png'
FIG_DPI: Final[int] = 600

# The boolean value to switch whether to display the value of the
# magnetic Ekman number or not when E_ETA = 0
SWITCH_DISP_ETA: Final[bool] = False

# ================================ #

SIZE_SUBMAT: Final[int] = N_T + 1

LINSP_PHI: Final[ArrayFloat] = np.linspace(0, 2*np.pi, NUM_PHI)

TEXT_TITLE: Final[str] = r'$|\alpha|=$' + f' {ALPHA}, ' \
    + r'$E_\eta=$' + f' {E_ETA}, ' + r'$R=$' + f' {ROSSBY}' \
    if SWITCH_DISP_ETA \
    else r'$|\alpha|=$' + f' {ALPHA}, ' + r'$R=$' + f' {ROSSBY}'


def wrapper_plot(result: DictLinsimResults) -> None:
    """Plot figures of the simulation results.

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.
    """

    invexpan_mat: ArrayFloat = create_cheb_invexpan_mat(SIZE_SUBMAT)

    num_step: int = len(result['time'])
    progress_bar: ProgressBar \
        = create_function_name_progress_bar(num_step)
    progress_bar.start()
    for i_step in range(num_step):

        plot_map(result, i_step, invexpan_mat)

        progress_bar.update(i_step)


def plot_map(result: DictLinsimResults,
             i_step: int,
             invexpan_mat: ArrayFloat) -> None:
    """Plot figures of the fields (2D contour map).

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.
    i_step : int
        The current time step.
    invexpan_mat : ArrayFloat
        The inverse Chebyshev expansion matrix.
    """

    m_orders: ArrayInt = result['m_orders']
    linsp_mu: ArrayFloat = result['linsp_mu']
    time: float = result['time'][i_step]
    array_psi: ArrayComplex = result['psi'][:, i_step]
    array_mvp: ArrayComplex = result['mvp'][:, i_step]
    bf_u_sin: ArrayFloat = result['bf_u_sin'][i_step]
    bf_b_sin: ArrayFloat = result['bf_b_sin'][i_step]

    linsp_theta: Final[ArrayFloat] = np.arccos(linsp_mu)

    grid_phi: ArrayFloat
    grid_theta: ArrayFloat
    grid_phi, grid_theta = np.meshgrid(LINSP_PHI, linsp_theta)

    grid_lat: Final[ArrayFloat] = np.rad2deg(
        np.full_like(grid_theta, np.pi/2) - grid_theta)
    grid_lon: Final[ArrayFloat] = np.rad2deg(grid_phi)

    psi: ArrayComplex
    mvp: ArrayComplex
    phase: ArrayComplex
    psi_grid: ArrayFloat = np.zeros_like(grid_phi)
    mvp_grid: ArrayFloat = np.zeros_like(grid_phi)
    for i_m, m_order in enumerate(m_orders):
        psi = cast(ArrayComplex, invexpan_mat @ array_psi[i_m])
        mvp = cast(ArrayComplex, invexpan_mat @ array_mvp[i_m])
        psi *= (1-linsp_mu**2)
        mvp *= (1-linsp_mu**2)

        phase = np.cos(m_order * grid_phi) + 1j*np.sin(m_order * grid_phi)
        psi_grid += np.real(psi[:, np.newaxis] * phase)
        mvp_grid += np.real(mvp[:, np.newaxis] * phase)

    plotter: DefaultGridPlotter = create_plotter(
        1, 2, figsize=(10, 5),
        subplot_kw={'projection':
                    ccrs.Mollweide(central_longitude=0.0)})

    level_psi: ArrayFloat = np.arange(-MAX_PSI, 1.2*MAX_PSI, 0.2*MAX_PSI)
    level_mvp: ArrayFloat = np.arange(-MAX_MVP, 1.2*MAX_MVP, 0.2*MAX_MVP)

    contour1: QuadContourSet = plotter.axes[0].contourf(
        grid_lon, grid_lat, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(),
        vmin=-MAX_PSI, vmax=MAX_PSI, cmap='bwr_r', extend='both')
    plotter.axes[0].contour(
        grid_lon, grid_lat, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(), colors='k', linewidths=0.8)

    contour2: QuadContourSet = plotter.axes[1].contourf(
        grid_lon, grid_lat, mvp_grid, levels=level_mvp,
        transform=ccrs.PlateCarree(),
        vmin=-MAX_MVP, vmax=MAX_MVP, cmap='PiYG_r', extend='both')
    plotter.axes[1].contour(
        grid_lon, grid_lat, mvp_grid, levels=level_mvp,
        transform=ccrs.PlateCarree(), colors='k', linewidths=0.8)

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
        r'vector potential $\mathrm{sgn}(\alpha)_1'
        + r'/\sqrt{\rho_0\mu_\mathrm{m}}$', fontsize=16)

    plotter.fig.suptitle(
        TEXT_TITLE + '\n\n' +
        r'$\tau=2\Omega_0 t=$' + f' {time:7.3f}',
        fontsize=16)

    name_fig: str = NAME_FIG[0] + f'_{i_step:0>6}' + NAME_FIG_SUFFIX
    plotter.save(PATH_DIR_FIG, name_fig, FIG_DPI, mute_log_message=True)
    plotter.close()


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    logger: DefaultLogger = DefaultLogger(__name__)

    logger.show_params(f'{ID=}',
                       f'{ALPHA=}',
                       f'{E_ETA=}',
                       f'{ROSSBY=}',
                       f'{N_T=}')

    data: DictLinsimResults \
        = load_linsim_results(PATH_DIR_INPUT, NAME_FILE)

    wrapper_plot(data)

    
