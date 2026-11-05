"""A Python script to plot the results of time-dependent (quasi-)linear
simulations for two-dimensional (2D) incompressible magnetohydrodynamic (MHD)
waves on a rotating sphere under a toroidal background field, B_phi = B_0
B(theta) sin(theta), and a background zonal flow, U_phi = U_0 U(theta)
sin(theta).

This script creates three kinds of figures: 2D contour maps, a time series
plot, and a Hovmoller diagram.

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
    $ uv run python mhd2dsphere_linsimfig.py
Run the script with a specified value of `ID` (say 0001):
    $ uv run python mhd2dsphere_linsimfig.py 0001
"""

from pathlib import Path

import cartopy.crs as ccrs
import numpy as np

from package_common.calc_heinrichs import heinrichs
from package_common.common_types import (ArrayComplex, ArrayFloat, ArrayInt,
                                         Final, FuncFloat, cast)
from package_common.default_logger import DefaultLogger
from package_common.default_plotter import (Colorbar, DefaultGridPlotter,
                                            DefaultPlotter, QuadContourSet,
                                            create_plotter)
from package_common.default_timer import DefaultTimer
from package_common.progress_bar import ProgressBar
from package_common.spectral_deform import init_complex_coordinate_standard
from package_common.utils_collocation import (ChebyshevGaussQuad, Func4Quad,
                                              create_cheb_diff_mat,
                                              create_cheb_expan_mat,
                                              create_cheb_invexpan_mat,
                                              spherical_laplacian_heinrichs)
from package_common.utils_input import input_string
from package_common.utils_name import create_function_name_progress_bar
from package_mhd2dsphere.load_data import load_linsim_results
from package_mhd2dsphere.typed_dict import (DictChebyshevGaussQuad,
                                            DictLinsimResults)

# ========== Parameters ========== #

# the run index
ID: Final[str] = input_string('0000')

# The boolean values to switch whether to plot figures or not
# SWITCH_PLOT[0]: 2D contour maps
# SWITCH_PLOT[1]: time series plot
# SWITCH_PLOT[2]: Hovmoller diagram
SWITCH_PLOT: Final[tuple[bool, bool, bool]] = (True, True, True)

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

# The colatitude for the Hovmoller diagram
THETA_HOVMOLLER: Final[float] = 90

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
NAME_FIG: Final[tuple[str, str, str]] = ('2d', 'time_series', 'hovmoller')
NAME_FIG_SUFFIX: Final[str] = '.png'
FIG_DPI: Final[int] = 600

# The boolean value to switch whether to display the value of the
# magnetic Ekman number or not when E_ETA = 0
SWITCH_DISP_ETA: Final[bool] = True

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

    if SWITCH_PLOT[0]:
        invexpan_mat: ArrayFloat \
            = create_cheb_invexpan_mat(SIZE_SUBMAT+2)[1:-1, :SIZE_SUBMAT]

        num_snapshot: int = len(result['time'])
        progress_bar: ProgressBar \
            = create_function_name_progress_bar(num_snapshot)
        progress_bar.start()
        for i_snapshot in range(num_snapshot):
            plot_map(result, i_snapshot, invexpan_mat)

            progress_bar.update(i_snapshot)

    if SWITCH_PLOT[1]:
        plot_time_series(result)

    if SWITCH_PLOT[2]:
        plot_hovmoller(result)


def plot_map(result: DictLinsimResults,
             i_snapshot: int,
             invexpan_mat: ArrayFloat) -> None:
    """Plot figures of the fields (2D contour map).

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.
    i_snapshot : int
        The current time step.
    invexpan_mat : ArrayFloat
        The inverse Chebyshev expansion matrix.
    """

    linsp_mu: ArrayFloat = result['linsp_mu']
    time: float = result['time'][i_snapshot]

    linsp_theta: Final[ArrayFloat] = np.arccos(linsp_mu)

    grid_phi: ArrayFloat
    grid_theta: ArrayFloat
    grid_phi, grid_theta = np.meshgrid(LINSP_PHI, linsp_theta)

    grid_lat: Final[ArrayFloat] = np.rad2deg(
        np.full_like(grid_theta, np.pi/2) - grid_theta)
    grid_lon: Final[ArrayFloat] = np.rad2deg(grid_phi)

    psi_grid: ArrayFloat
    mvp_grid: ArrayFloat
    psi_grid, mvp_grid = create_profile(result, i_snapshot, invexpan_mat)

    plotter: DefaultGridPlotter = create_plotter(
        1, 2, figsize=(10, 5),
        subplot_kw={'projection':
                    ccrs.Mollweide(central_longitude=0.0)})

    max_psi: float = np.nanmax(np.abs(result['psi']))
    max_mvp: float = np.nanmax(np.abs(result['mvp']))

    level_psi: ArrayFloat = np.arange(-max_psi, 1.2*max_psi, 0.2*max_psi)
    level_mvp: ArrayFloat
    if max_mvp != 0:
        level_mvp = np.arange(-max_mvp, 1.2*max_mvp, 0.2*max_mvp)
    else:
        level_mvp = np.arange(-1, 1.2, 0.2)

    contour1: QuadContourSet = plotter.axes[0].contourf(
        grid_lon, grid_lat, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(),
        vmin=-max_psi, vmax=max_psi, cmap='bwr_r', extend='both')
    plotter.axes[0].contour(
        grid_lon, grid_lat, psi_grid, levels=level_psi,
        transform=ccrs.PlateCarree(), colors='k', linewidths=0.8)

    contour2: QuadContourSet = plotter.axes[1].contourf(
        grid_lon, grid_lat, mvp_grid, levels=level_mvp,
        transform=ccrs.PlateCarree(),
        vmin=-max_mvp, vmax=max_mvp, cmap='PiYG_r', extend='both')
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
        r'vector potential $\mathrm{sgn}(\alpha)a_1'
        + r'/\sqrt{\rho_0\mu_\mathrm{m}}$', fontsize=16)

    plotter.fig.suptitle(
        TEXT_TITLE + '\n\n' +
        r'$\tau=2\Omega_0 t=$' + f' {time:7.3f}',
        fontsize=16)

    name_fig: str = NAME_FIG[0] + f'_{i_snapshot:0>6}' + NAME_FIG_SUFFIX
    plotter.save(PATH_DIR_FIG, name_fig, FIG_DPI, mute_log_message=True)
    plotter.close()


def create_profile(result: DictLinsimResults,
                   i_snapshot: int,
                   invexpan_mat: ArrayFloat) -> tuple[ArrayFloat,
                                                      ArrayFloat]:
    """Create meshgrids of the stream function and vector potential.

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.
    i_snapshot : int
        The current time step.
    invexpan_mat : ArrayFloat
        The inverse expansion matrix.

    Returns
    -------
    psi_grid : ArrayFloat
        The meshgrid of the stream function.
    mvp_grid : ArrayFloat
        The meshgrid of the vector potential.
    """

    m_orders: ArrayInt = result['m_orders']
    linsp_mu: ArrayFloat = result['linsp_mu']
    array_psi: ArrayComplex = result['psi'][:, i_snapshot]
    array_mvp: ArrayComplex = result['mvp'][:, i_snapshot]

    linsp_theta: Final[ArrayFloat] = np.arccos(linsp_mu)

    grid_phi: ArrayFloat
    grid_phi, _ = np.meshgrid(LINSP_PHI, linsp_theta)

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

    return psi_grid, mvp_grid


def plot_time_series(result: DictLinsimResults) -> None:
    """Plot the time series of the simulation results.

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.
    """

    time: ArrayFloat = result['time']

    pke, pme, psm, pse, ohm = calc_qty(result)

    norm_ene: float = (pke + pme)[0]
    norm_psm: float = psm[0]
    pke /= norm_ene
    pme /= norm_ene
    if norm_psm != 0:
        psm /= norm_psm
    pse /= norm_ene

    plotter: DefaultPlotter = create_plotter(1, 1, figsize=(5, 5))

    plotter.axes.plot(time, pke+pme, color='red', linestyle='-.',
                      label='perturbation energy')
    plotter.axes.plot(time, psm, color='blue',
                      label='angular pseudomomentum')
    plotter.axes.plot(time, pse, color='red', linestyle='-',
                      label='pseudoenergy')
    if E_ETA != 0:
        ohm /= norm_ene
        plotter.axes.plot(time, ohm, color='green',
                          label='ohmic dissipation')

    plotter.axes.set_xlabel(r'$\tau=2\Omega_0 t$', fontsize=16)
    plotter.axes.set_title(TEXT_TITLE, fontsize=16)

    plotter.leg = plotter.axes.legend(loc='best', fontsize=12)

    plotter.axes.tick_params(labelsize=14)

    name_fig: str = NAME_FIG[1] + NAME_FIG_SUFFIX
    plotter.save(PATH_DIR_FIG, name_fig, FIG_DPI)
    plotter.close()


def calc_qty(result: DictLinsimResults) -> tuple[ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayFloat]:
    """Calculate various quantities from the simulation results.

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.

    Returns
    -------
    pke : ArrayFloat
        The perturbation kinetic energy.
    pme : ArrayFloat
        The perturbation magnetic energy.
    psm : ArrayFloat
        Angular pseudomomentum.
    pse : ArrayFloat
        Pseudoenergy.
    ohm : ArrayFloat
        Ohmic dissipation.
    """

    m_orders: ArrayInt = result['m_orders']
    time: ArrayFloat = result['time']
    array_psi: ArrayComplex = result['psi']
    array_mvp: ArrayComplex = result['mvp']

    pke: ArrayFloat = np.zeros_like(time)
    pme: ArrayFloat = np.zeros_like(time)
    psm: ArrayFloat = np.zeros_like(time)
    pse: ArrayFloat = np.zeros_like(time)
    ohm: ArrayFloat = np.zeros_like(time)

    diff_mat: ArrayFloat = create_cheb_diff_mat(SIZE_SUBMAT+2)
    expan_mat: ArrayFloat = create_cheb_expan_mat(SIZE_SUBMAT+2)

    ChebyshevGaussQuad.set_class_variable(
        SIZE_SUBMAT, y_complex=init_complex_coordinate_standard(-1, 1))

    vec_psi: ArrayComplex
    vec_mvp: ArrayComplex
    psm_1: ArrayComplex
    psm_2: ArrayComplex
    pse_u1: ArrayComplex
    pse_u2: ArrayComplex
    pse_b: ArrayComplex

    for i_m, m_order in enumerate(m_orders):

        def _minus_spherical_laplacian_heinrichs(n_degree: int,
                                                 mu: float) -> float:
            return (-1) \
                * cast(float,
                       spherical_laplacian_heinrichs(m_order, n_degree, mu))

        for i_snapshot in range(len(result['time'])):

            vec_psi = array_psi[i_m, i_snapshot]
            vec_mvp = array_mvp[i_m, i_snapshot]

            dict_quad: DictChebyshevGaussQuad \
                = prepare_chebyshev_gauss_quad(
                    result, i_snapshot, expan_mat, diff_mat,
                    cast(Func4Quad, _minus_spherical_laplacian_heinrichs))

            pke[i_snapshot] += np.real(dict_quad['quad_pke'].quadrature(
                vec_1=vec_psi, vec_2=vec_psi))
            pme[i_snapshot] += np.real(dict_quad['quad_pme'].quadrature(
                vec_1=vec_mvp, vec_2=vec_mvp))

            if ALPHA == 0:
                psm[i_snapshot] += np.real(dict_quad['quad_psm_hd'].quadrature(
                    vec_1=vec_psi, vec_2=vec_psi))
                pse[i_snapshot] += np.real(dict_quad['quad_pse_hd'].quadrature(
                    vec_1=vec_psi, vec_2=vec_psi))
            else:
                psm_1 = dict_quad['quad_psm_1'].quadrature(
                    vec_1=vec_psi, vec_2=vec_mvp) / ALPHA
                psm_2 = dict_quad['quad_psm_2'].quadrature(
                    vec_1=vec_mvp, vec_2=vec_mvp) / (ALPHA**2)
                psm[i_snapshot] += np.real(psm_1 + np.conj(psm_1) + psm_2)

                pse_u1 = dict_quad['quad_pse_u1'].quadrature(
                    vec_1=vec_psi, vec_2=vec_mvp) / ALPHA
                pse_u2 = dict_quad['quad_pse_u2'].quadrature(
                    vec_1=vec_mvp, vec_2=vec_mvp) / (ALPHA**2)
                pse_b = dict_quad['quad_pse_b'].quadrature(
                    vec_1=vec_mvp, vec_2=vec_mvp)
                pse[i_snapshot] \
                    += np.real(pse_u1 + np.conj(pse_u1) + pse_u2 + pse_b)

            ohm[i_snapshot] += np.real(dict_quad['quad_ohm'].quadrature(
                vec_1=vec_mvp, vec_2=vec_mvp))

    pse += pke
    if ALPHA != 0:
        pse += pme

    ohm *= E_ETA

    return pke, pme, psm, pse, ohm


def prepare_chebyshev_gauss_quad(
        result: DictLinsimResults,
        i_snapshot: int,
        expan_mat: ArrayFloat,
        diff_mat: ArrayFloat,
        minus_spherical_laplacian_heinrichs: Func4Quad) \
        -> DictChebyshevGaussQuad:
    """Prepare the Chebyshev-Gauss quadrature.

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.
    i_snapshot : int
        The current time step.
    expan_mat : ArrayFloat
        The Chebyshev expansion matrix.
    diff_mat : ArrayFloat
        The Chebyshev differentiation matrix.
    minus_spherical_laplacian_heinrichs : Func4Quad
        The negative of the spherical Laplacian of Heinrichs bases.

    Returns
    -------
    quad : DictChebyshevGaussQuad
        The dictionary for the Chebyshev-Gauss quadrature.
    """

    linsp_mu: ArrayFloat = result['linsp_mu']
    bf_u: ArrayFloat = result['bf_u'][i_snapshot]
    bf_b: ArrayFloat = result['bf_b'][i_snapshot]

    sin2: ArrayFloat = 1-(linsp_mu**2)
    bf_u_sin2: ArrayFloat = bf_u * sin2
    bf_u_shear: ArrayFloat \
        = diff_mat @ (diff_mat @ np.concatenate(([0], bf_u_sin2, [0])))
    bf_b_sin2: ArrayFloat = bf_b * sin2
    bf_b_shear: ArrayFloat \
        = diff_mat @ (diff_mat @ np.concatenate(([0], bf_b_sin2, [0])))

    vec_u_sin2: ArrayFloat = expan_mat @ np.concatenate(([0], bf_u_sin2, [0]))
    vec_b_sin2: ArrayFloat = expan_mat @ np.concatenate(([0], bf_b_sin2, [0]))
    vec_u_shear: ArrayFloat = expan_mat @ bf_u_shear
    vec_b_shear: ArrayFloat = expan_mat @ bf_b_shear

    _bf_u_sin2: FuncFloat = cast(
        FuncFloat, ChebyshevGaussQuad.cheb_coeff2func(vec_u_sin2))
    _bf_b_sin2: FuncFloat = cast(
        FuncFloat, ChebyshevGaussQuad.cheb_coeff2func(vec_b_sin2))
    _bf_u_shear: FuncFloat = cast(
        FuncFloat, ChebyshevGaussQuad.cheb_coeff2func(vec_u_shear))
    _bf_b_shear: FuncFloat = cast(
        FuncFloat, ChebyshevGaussQuad.cheb_coeff2func(vec_b_shear))

    def _weight_psm_1(mu: float) -> float:
        b_mu: float = _bf_b_sin2(mu) / (1-(mu**2))
        return (-1) / (4 * b_mu)

    def _weight_psm_2(mu: float) -> float:
        b_mu: float = _bf_b_sin2(mu) / (1-(mu**2))
        u_shear_mu: float = _bf_u_shear(mu)
        return (-1) * (ROSSBY*u_shear_mu-1) / (4 * (b_mu**2))

    def _weight_pse_u1(mu: float) -> float:
        u_mu: float = _bf_u_sin2(mu) / (1-(mu**2))
        return 4 * ROSSBY * u_mu * _weight_psm_1(mu)

    def _weight_pse_u2(mu: float) -> float:
        u_mu: float = _bf_u_sin2(mu) / (1-(mu**2))
        return 4 * ROSSBY * u_mu * _weight_psm_2(mu)

    def _weight_pse_b(mu: float) -> float:
        b_mu: float = _bf_b_sin2(mu) / (1-(mu**2))
        b_shear_mu: float = _bf_b_shear(mu)
        return b_shear_mu / b_mu

    def _weight_psm_hd(mu: float) -> float:
        u_shear_mu: float = _bf_u_shear(mu)
        return 1 / (4 * (ROSSBY*u_shear_mu-1))

    def _weight_pse_hd(mu: float) -> float:
        u_mu: float = _bf_u_sin2(mu) / (1-(mu**2))
        return 4 * ROSSBY * u_mu * _weight_psm_hd(mu)

    quad: DictChebyshevGaussQuad = {
        'quad_pke': ChebyshevGaussQuad(
            func_1=heinrichs,
            func_2=minus_spherical_laplacian_heinrichs
        ),
        'quad_pme': ChebyshevGaussQuad(
            func_1=heinrichs,
            func_2=minus_spherical_laplacian_heinrichs
        ),
        'quad_psm_1': ChebyshevGaussQuad(
            func_1=minus_spherical_laplacian_heinrichs,
            func_2=heinrichs,
            weight=_weight_psm_1
        ),
        'quad_psm_2': ChebyshevGaussQuad(
            func_1=heinrichs,
            func_2=heinrichs,
            weight=_weight_psm_2
        ),
        'quad_pse_u1': ChebyshevGaussQuad(
            func_1=minus_spherical_laplacian_heinrichs,
            func_2=heinrichs,
            weight=_weight_pse_u1
        ),
        'quad_pse_u2': ChebyshevGaussQuad(
            func_1=heinrichs,
            func_2=heinrichs,
            weight=_weight_pse_u2
        ),
        'quad_pse_b': ChebyshevGaussQuad(
            func_1=heinrichs,
            func_2=heinrichs,
            weight=_weight_pse_b
        ),
        'quad_ohm': ChebyshevGaussQuad(
            func_1=minus_spherical_laplacian_heinrichs,
            func_2=minus_spherical_laplacian_heinrichs
        ),
        'quad_psm_hd': ChebyshevGaussQuad(
            func_1=minus_spherical_laplacian_heinrichs,
            func_2=minus_spherical_laplacian_heinrichs,
            weight=_weight_psm_hd
        ),
        'quad_pse_hd': ChebyshevGaussQuad(
            func_1=minus_spherical_laplacian_heinrichs,
            func_2=minus_spherical_laplacian_heinrichs,
            weight=_weight_pse_hd
        ),
    }

    return quad


def plot_hovmoller(result: DictLinsimResults) -> None:
    """Plot the Hovmoller diagram.

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.
    """

    time: ArrayFloat = result['time']

    grid_phi: ArrayFloat
    grid_time: ArrayFloat
    grid_phi, grid_time = np.meshgrid(LINSP_PHI, time)

    psi_hovmoller: ArrayFloat
    mvp_hovmoller: ArrayFloat
    psi_hovmoller, mvp_hovmoller = create_hovmoller(result)

    plotter: DefaultGridPlotter = create_plotter(1, 2, figsize=(10, 5))

    max_psi: float = np.nanmax(np.abs(psi_hovmoller))
    max_mvp: float = np.nanmax(np.abs(mvp_hovmoller))

    level_psi: ArrayFloat = np.arange(-max_psi, 1.2*max_psi, 0.2*max_psi)
    level_mvp: ArrayFloat
    if max_mvp != 0:
        level_mvp = np.arange(-max_mvp, 1.2*max_mvp, 0.2*max_mvp)
    else:
        level_mvp = np.arange(-1, 1.2, 0.2)

    contour1: QuadContourSet = plotter.axes[0].contourf(
        grid_phi, grid_time, psi_hovmoller, levels=level_psi,
        vmin=-max_psi, vmax=max_psi, cmap='bwr_r')
    plotter.axes[0].contour(
        grid_phi, grid_time, psi_hovmoller, levels=level_psi,
        colors='k', linewidths=0.3)

    contour2: QuadContourSet = plotter.axes[1].contourf(
        grid_phi, grid_time, mvp_hovmoller, levels=level_mvp,
        vmin=-max_mvp, vmax=max_mvp, cmap='PiYG_r')
    plotter.axes[1].contour(
        grid_phi, grid_time, mvp_hovmoller, levels=level_mvp,
        colors='k', linewidths=0.3)

    plotter.axes[0].grid(linestyle=':')
    plotter.axes[1].grid(linestyle=':')

    cbar1: Colorbar = plotter.fig.colorbar(contour1, ax=plotter.axes[0])
    cbar2: Colorbar = plotter.fig.colorbar(contour2, ax=plotter.axes[1])
    cbar1.ax.tick_params(labelsize=14)
    cbar2.ax.tick_params(labelsize=14)

    for axis in (plotter.axes[0], plotter.axes[1]):
        axis.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        axis.set_xticklabels(['$0$', '$90$', '$180$', '$270$', '$360$'])
        axis.set_xlabel('longitude [degree]', fontsize=14)
        axis.set_ylabel(r'$\tau=2\Omega_0 t$', fontsize=14)

    plotter.axes[0].set_title(r'stream function $\psi_1$', fontsize=16)
    plotter.axes[1].set_title(
        r'vector potential $\mathrm{sgn}(\alpha)a_1'
        + r'/\sqrt{\rho_0\mu_\mathrm{m}}$', fontsize=16)

    plotter.fig.suptitle(
        TEXT_TITLE + f', '
        + r'$\theta=$' + f' {THETA_HOVMOLLER}' + r'$^\circ$', fontsize=16)

    name_fig: str = NAME_FIG[2] + NAME_FIG_SUFFIX
    plotter.save(PATH_DIR_FIG, name_fig, FIG_DPI)
    plotter.close()


def create_hovmoller(result: DictLinsimResults) -> tuple[ArrayFloat,
                                                         ArrayFloat]:
    """Create the Hovmoller diagram.

    Parameters
    ----------
    result : DictLinsimResults
        The dictionary of the simulation results.

    Returns
    -------
    psi_hovmoller : ArrayFloat
        The Hovmoller diagram of the stream function.
    mvp_hovmoller : ArrayFloat
        The Hovmoller diagram of the vector potential.
    """

    linsp_mu: ArrayFloat = result['linsp_mu']
    time: ArrayFloat = result['time']

    mu_hovmoller: float = np.cos(np.deg2rad(THETA_HOVMOLLER))
    i_hovmoller: int = int(np.argmin(np.abs(linsp_mu - mu_hovmoller)))

    invexpan_mat: ArrayFloat \
        = create_cheb_invexpan_mat(SIZE_SUBMAT+2)[1:-1, :SIZE_SUBMAT]

    grid_phi: ArrayFloat
    grid_phi, _ = np.meshgrid(LINSP_PHI, time)

    psi_grid: ArrayFloat
    mvp_grid: ArrayFloat
    psi_hovmoller: ArrayFloat = np.zeros_like(grid_phi)
    mvp_hovmoller: ArrayFloat = np.zeros_like(grid_phi)

    for i_snapshot in range(len(result['time'])):
        psi_grid, mvp_grid = create_profile(result, i_snapshot, invexpan_mat)

        psi_hovmoller[i_snapshot, :] = psi_grid[i_hovmoller, :]
        mvp_hovmoller[i_snapshot, :] = mvp_grid[i_hovmoller, :]

    return psi_hovmoller, mvp_hovmoller


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    logger: DefaultLogger = DefaultLogger(__name__)

    logger.show_params(f'{ID=}',
                       f'{ALPHA=}',
                       f'{E_ETA=}',
                       f'{ROSSBY=}',
                       f'{N_T=}',
                       f'{THETA_HOVMOLLER=}')

    data: DictLinsimResults \
        = load_linsim_results(PATH_DIR_INPUT, NAME_FILE)

    wrapper_plot(data)

    timer.end()
