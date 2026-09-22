"""A Python script to perform time-dependent (quasi-)linear simulations for
two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and a background zonal flow, U_phi = U_0 U(theta) sin(theta).

This script can create up to ... figures:

Warnings
--------

Notes
-----
All parameters are described within the script.

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
$ uv run python mhd2dsphere_linsim.py
"""

import time
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np

from package_common.background_field import BackgroundField
from package_common.common_types import (Any, ArrayComplex, ArrayFloat, Final,
                                         cast)
from package_common.default_logger import DefaultLogger
from package_common.default_plotter import (Colorbar, DefaultGridPlotter,
                                            QuadContourSet, create_plotter)
from package_common.default_timer import DefaultTimer
from package_common.progress_bar import ProgressBar
from package_common.utils_collocation import (calc_collocation_point,
                                              create_cheb_invexpan_mat)
from package_common.utils_name import create_function_name_progress_bar
from package_common.utils_simulation import Field, Rhs, time_integrate
from package_mhd2dsphere import (init_background_b, init_background_u,
                                 init_linsim)
from package_mhd2dsphere.create_rhs import wrapper_rhs

# ========== Parameters ========== #

# The zonal wavenumbers (orders)
M_ORDERS: Final[list[int]] = [2, ]

# The Lehnert number
ALPHA: Final[float] = 0.1

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The Rossby number
ROSSBY: Final[float] = 0

# The truncation degree
N_T: Final[int] = 500

# Time step
NUM_STEP: Final[int] = 10**4
DT: Final[float] = 1e-2
SAVE_INTERVAL: Final[int] = 100

# Initial conditions
INIT_BG_FIELD_B: Final[BackgroundField] = init_background_b.b_malkus('mu')
INIT_BG_FIELD_U: Final[BackgroundField] = init_background_u.u_rigid('mu')
DICT_INIT_PSI: Final[dict[int, ArrayFloat]] = {
    M_ORDERS[0]: init_linsim.init_spherical_harmonics(N_T+1, 2, M_ORDERS[0])
}
DICT_INIT_MVP: Final[dict[int, Any]] = {}

# The boolean value to switch whether to perform quasi-linear simulations or
# not.
SWITCH_QUASI_LIN: Final[bool] = False

# The number of grid points in the phi direction
NUM_PHI: Final[int] = 361

# The maximum values for the color scales
MAX_PSI: Final[float] = 5
MAX_MVP: Final[float] = 5

# The paths and filenames of outputs
NAME_DIR: Final[str] = (
    f'MHD2Dsphere_linsim'
    + f'_m={M_ORDERS}_a={ALPHA}_E={E_ETA}_R={ROSSBY}_N={N_T}'
    + f'_{time.time()}'
)
PATH_DIR: Final[Path] = Path('.') / 'fig' / 'MHD2Dsphere_linsim' / NAME_DIR
NAME_FIG_SUFFIX: Final[str] = '.png'
FIG_DPI: Final[int] = 600

# ================================ #

SIZE_SUBMAT: Final[int] = N_T + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

LINSP_MU: Final[ArrayFloat] = np.array([
    calc_collocation_point(i_l+1, SIZE_SUBMAT+2)
    for i_l in range(SIZE_SUBMAT)])
LINSP_THETA: Final[ArrayFloat] = np.arccos(LINSP_MU)
LINSP_PHI: Final[ArrayFloat] = np.linspace(0, 2*np.pi, NUM_PHI)

GRID_PHI: ArrayFloat
GRID_THETA: ArrayFloat
GRID_PHI, GRID_THETA = np.meshgrid(LINSP_PHI, LINSP_THETA)

GRID_LAT: Final[ArrayFloat] = np.rad2deg(
    np.full_like(GRID_THETA, np.pi/2) - GRID_THETA)
GRID_LON: Final[ArrayFloat] = np.rad2deg(GRID_PHI)

TEXT_TITLE: Final[str] = r'$|\alpha|=$' + f' {ALPHA}, ' \
    + r'$E_\eta=$' + f' {E_ETA}, ' + r'$R=$' + f' {ROSSBY}'


def initialize_fields(psi: list[Field],
                      mvp: list[Field],
                      bf_u_sin: Field,
                      bf_b_sin: Field) -> tuple[list[Field],
                                                list[Field],
                                                Field,
                                                Field]:
    """Initialize the values of the fields for the (quasi-)linear simulation.

    Parameters
    ----------
    psi : list[Field]
        The list of the stream function (psi).
    mvp : list[Field]
        The list of the vector potential (a).
    bf_u_sin : Field
        The background zonal flow.
    bf_b_sin : Field
        The toroidal background field.

    Returns
    -------
    tuple[list[Field], list[Field], Field, Field]
        The list of fields (list of psi, list of mvp, bf_u_sin, and bf_b_sin).
    """

    for i_m, m_order in enumerate(M_ORDERS):
        if m_order in DICT_INIT_PSI.keys():
            psi[i_m].value += DICT_INIT_PSI[m_order]
        if m_order in DICT_INIT_MVP.keys():
            mvp[i_m].value += DICT_INIT_MVP[m_order]

    bf_u_sin.value += np.array(
        [INIT_BG_FIELD_U.r_value(mu) * np.sqrt(1-mu**2) for mu in LINSP_MU])
    bf_b_sin.value += np.array(
        [INIT_BG_FIELD_B.r_value(mu) * np.sqrt(1-mu**2) for mu in LINSP_MU])

    return psi, mvp, bf_u_sin, bf_b_sin


def plot_map(fields: list[Field],
             i_step: int,
             inverpan_mat: ArrayFloat) -> None:
    """Plot figures of the fields (2D contour map).

    Parameters
    ----------
    fields : list[Field]
        The list of fields (list of psi, list of mvp, bf_u_sin, and bf_b_sin).
    i_step : int
        The current time step.
    inverpan_mat : ArrayFloat
        The inverse Chebyshev expansion matrix.
    """

    num_m: int = len(M_ORDERS)

    list_psi: list[Field] = fields[0:num_m]
    list_mvp: list[Field] = fields[num_m:2*num_m]
    bf_u_sin: Field = fields[2*num_m]
    bf_b_sin: Field = fields[2*num_m+1]

    psi: ArrayComplex
    mvp: ArrayComplex
    phase: ArrayComplex
    psi_grid: ArrayFloat = np.zeros_like(GRID_PHI)
    mvp_grid: ArrayFloat = np.zeros_like(GRID_PHI)
    for i_m, m_order in enumerate(M_ORDERS):
        psi = cast(ArrayComplex, inverpan_mat @ list_psi[i_m].value)
        mvp = cast(ArrayComplex, inverpan_mat @ list_mvp[i_m].value)
        psi *= (1-LINSP_MU**2)
        mvp *= (1-LINSP_MU**2)

        phase = np.cos(m_order * GRID_PHI) + 1j*np.sin(m_order * GRID_PHI)
        psi_grid += np.real(psi[:, np.newaxis] * phase)
        mvp_grid += np.real(mvp[:, np.newaxis] * phase)

    plotter: DefaultGridPlotter = create_plotter(
        1, 2, figsize=(10, 5),
        subplot_kw={'projection':
                    ccrs.Mollweide(central_longitude=0.0)})

    level_psi: ArrayFloat = np.arange(-MAX_PSI, 1.2*MAX_PSI, 0.2*MAX_PSI)
    level_mvp: ArrayFloat = np.arange(-MAX_MVP, 1.2*MAX_MVP, 0.2*MAX_MVP)

    contour1: QuadContourSet = plotter.axes[0].contourf(
        GRID_LON, GRID_LAT, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(),
        vmin=-MAX_PSI, vmax=MAX_PSI, cmap='bwr_r', extend='both')
    plotter.axes[0].contour(
        GRID_LON, GRID_LAT, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(), colors='k', linewidths=0.8)

    contour2: QuadContourSet = plotter.axes[1].contourf(
        GRID_LON, GRID_LAT, mvp_grid, levels=level_mvp,
        transform=ccrs.PlateCarree(),
        vmin=-MAX_MVP, vmax=MAX_MVP, cmap='PiYG_r', extend='both')
    plotter.axes[1].contour(
        GRID_LON, GRID_LAT, mvp_grid, levels=level_mvp,
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
        r'vector potential $\mathrm{sgn}(\alpha)a_1'
        + r'/\sqrt{\rho_0\mu_\mathrm{m}}$', fontsize=16)

    plotter.fig.suptitle(
        TEXT_TITLE + '\n\n' +
        r'$\tau=2\Omega_0 t=$' + f' {bf_u_sin.time:7.3f}',
        fontsize=16)

    name_fig: str = f'{i_step:0>6}' + NAME_FIG_SUFFIX
    plotter.save(PATH_DIR, name_fig, FIG_DPI, mute_log_message=True)
    plotter.close()


def wrapper_time_integrate(psi: list[Field],
                           mvp: list[Field],
                           bf_u_sin: Field,
                           bf_b_sin: Field) -> None:
    """Perform time integration for the (quasi-)linear simulation.

    Parameters
    ----------
    psi : list[Field]
        The list of the stream function (psi).
    mvp : list[Field]
        The list of the vector potential (a).
    bf_u_sin : Field
        The background zonal flow.
    bf_b_sin : Field
        The toroidal background field.
    """

    fields: list[Field] = [*psi, *mvp, bf_u_sin, bf_b_sin]
    rhss: list[Rhs] = [
        *[wrapper_rhs('psi', m_order,
                      list_m_order=M_ORDERS, linsp_mu=LINSP_MU,
                      alpha=ALPHA, e_eta=E_ETA, rossby=ROSSBY,
                      switch_quasi_lin=SWITCH_QUASI_LIN)
          for m_order in M_ORDERS],
        *[wrapper_rhs('mvp', m_order,
                      list_m_order=M_ORDERS, linsp_mu=LINSP_MU,
                      alpha=ALPHA, e_eta=E_ETA, rossby=ROSSBY,
                      switch_quasi_lin=SWITCH_QUASI_LIN)
          for m_order in M_ORDERS],
        wrapper_rhs('bf_u_sin', list_m_order=M_ORDERS, linsp_mu=LINSP_MU,
                    alpha=ALPHA, e_eta=E_ETA, rossby=ROSSBY,
                    switch_quasi_lin=SWITCH_QUASI_LIN),
        wrapper_rhs('bf_b_sin', list_m_order=M_ORDERS, linsp_mu=LINSP_MU,
                    alpha=ALPHA, e_eta=E_ETA, rossby=ROSSBY,
                    switch_quasi_lin=SWITCH_QUASI_LIN)
    ]

    inverpan_mat: ArrayFloat = create_cheb_invexpan_mat(SIZE_SUBMAT)

    progress_bar: ProgressBar \
        = create_function_name_progress_bar(NUM_STEP)
    progress_bar.start()
    for i_step in range(NUM_STEP):

        time_integrate(fields, DT, rhss)

        if i_step % SAVE_INTERVAL == 0:
            plot_map(fields, i_step, inverpan_mat)

        progress_bar.update(i_step)


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    logger: DefaultLogger = DefaultLogger(__name__)

    logger.show_params(f'{M_ORDERS=}',
                       f'{ALPHA=}',
                       f'{E_ETA=}',
                       f'{ROSSBY=}',
                       f'{N_T=}',
                       f'{NUM_STEP=}',
                       f'{DT=}',
                       f'{SWITCH_QUASI_LIN=}')

    Field.set_class_variable(SIZE_SUBMAT)
    psi: list[Field] = [Field('psi', dtype=np.complex128) for _ in M_ORDERS]
    mvp: list[Field] = [Field('mvp', dtype=np.complex128) for _ in M_ORDERS]
    bf_u_sin: Field = Field('bf_u_sin')
    bf_b_sin: Field = Field('bf_b_sin')

    psi, mvp, bf_u_sin, bf_b_sin = initialize_fields(
        psi, mvp, bf_u_sin, bf_b_sin)

    wrapper_time_integrate(psi, mvp, bf_u_sin, bf_b_sin)

    timer.end()
