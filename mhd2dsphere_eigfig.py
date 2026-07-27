"""A Python script to plot the dispersion diagram of
two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and a background zonal flow, U_phi = U_0 U(theta) sin(theta).

This script can create up to four figures: linear-linear and log-log plots of
the dispersion relation with some coloring based on black or physical quantity
(energy partitioning, ohmic dissipation, pseudomomentum, pseudoenergy, or
eigenfrequencies). The script can also create a plot of the dispersion diagram
for a chosen alpha.

Parameters
----------
M_ORDER : int
    The zonal wavenumber (order).

Warnings
--------
No plotted figures
    If all of the boolean values to switch whether to plot figures or not are
    False.
Invalid value for 'SWITCH_COLOR'
    If 'SWITCH_COLOR' is not either 'blk', 'ene', 'ohm', 'psm', 'pse', or
    'qmode'.
Meaningless figures are plotted
    If figures of the ohmic dissipation are plotted in the ideal MHD case, and
    so on.

Notes
-----
All other parameters aside from command line arguments are described within the
script. Before executing this code, mhd2dsphere_eig.py with the same parameters
must be run.

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
Run the script with the default value of M_ORDER:
    $ python3 mhd2dsphere_eigfig.py
Run the script with a specified value (say M_ORDER = 2):
    $ python3 mhd2dsphere_eigfig.py 2
"""

import multiprocessing
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from scipy.linalg import svdvals

from package_common.background_field import BackgroundField
from package_common.common_types import (ArrayComplex, ArrayFloat, ArrayStr,
                                         Final, cast)
from package_common.default_logger import DefaultLogger
from package_common.default_plotter import (Axes, Colorbar, DefaultGridPlotter,
                                            DefaultPlotter, QuadContourSet,
                                            create_plotter)
from package_common.default_timer import DefaultTimer
from package_common.progress_bar import ProgressBar
from package_common.spectral_deform import (ComplexCoordinate,
                                            init_complex_coordinate_simple)
from package_common.utils_collocation import calc_collocation_point
from package_common.utils_input import input_value
from package_common.utils_name import create_function_name_progress_bar
from package_common.utils_parallel import (SharedInfo, SharedMemory,
                                           attach_shared_arrays,
                                           create_shared_arrays,
                                           detach_shared_arrays,
                                           set_num_process, set_num_threads)
from package_mhd2dsphere import init_background_b, init_background_u
from package_mhd2dsphere.create_mat import create_mat, create_submat
from package_mhd2dsphere.load_data import wrapper_load_results
from package_mhd2dsphere.processing_results import (pickup_eig, pickup_param,
                                                    screening_eig_q)
from package_mhd2dsphere.typed_dict import (DictBackgroundField, DictFileInfo,
                                            DictParams, DictResult)

type Bbox = transforms.Bbox

# ========== Parameters ========== #

# The boolean values to switch whether to plot figures or not
# SWITCH_PLOT[0]: The dispersion diagram for the linear-linear plot
# SWITCH_PLOT[1]: The dispersion diagram for the log-log plot
# SWITCH_PLOT[2]: The dispersion diagram for a chosen alpha
SWITCH_PLOT: Final[tuple[bool, bool, bool]] = (True, True, False)
ALPHA_CHOSEN: Final[float] = 1

# The coloring rule
# SWITCH_COLOR == 'blk': black
# SWITCH_COLOR == 'ene': energy partitioning
# SWITCH_COLOR == 'psm': pseudomomentum
# SWITCH_COLOR == 'pse': pseudoenergy
# SWITCH_COLOR == 'ohm': ohmic dissipation
# SWITCH_COLOR == 'qmode': for finding quasi-modes
SWITCH_COLOR: Final[str] = 'ene'

# Background field
BG_FIELD_B: Final[BackgroundField] = init_background_b.b_malkus('mu')
BG_FIELD_U: Final[BackgroundField] = init_background_u.u_rigid('mu')
# For the spectral deformation method
MU_COMPLEX: Final[ComplexCoordinate] = init_complex_coordinate_simple(
    -1, 1, alpha=0, beta_0=0, beta_1=0)
MU_COMPLEX_UNUSE_SPECTRAL_DEFORM: Final[ComplexCoordinate] \
    = init_complex_coordinate_simple(-1, 1)
# The boolean value to switch whether to follow Nakashima & Yoshida
# (2024)[1]_ or not
# If SWITCH_NY24 is True, BG_FIELD_B, BG_FIELD_U and
# MU_COMPLEX are ignored.
SWITCH_NY24: Final[bool] = False

# The zonal wavenumber (order)
M_ORDER: Final[int] = input_value(1, int)

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The Rossby number
ROSSBY: Final[float] = 0

# The truncation degree
N_T: Final[int] = 500
# N_T: Final[int] = 2000

# The criterion for plotting, which is based on the quality factor
CRITERION_Q: Final[float] = 0

# The range of eigenvalues
# linear, real part
EIG_RE_INIT: Final[float] = -2
EIG_RE_END: Final[float] = 2
# log, real part
EIG_RE_LOG_INIT: Final[float] = 10**(-6)
EIG_RE_LOG_END: Final[float] = 10**2
# linear & log, imaginary part
EIG_IM_LOG_MIN: Final[float] = 10**(-10)

# The number of grid points for epsilon-pseudospectrum
NUM_EIG_GRID: Final[int] = 100

# The paths and filenames of inputs
PATH_DIR_INPUT: Final[Path] = Path('.') / 'output' / 'MHD2Dsphere_eig'
NAME_FILE: Final[str] \
    = f'MHD2Dsphere_eig_NY24_m={M_ORDER}_E={E_ETA}_N={N_T}' \
    if SWITCH_NY24 \
    else f'MHD2Dsphere_eig_B{BG_FIELD_B.name}U{BG_FIELD_U.name}' \
    + f'_m={M_ORDER}_E={E_ETA}_R={ROSSBY}_N={N_T}' \
    + f'_{MU_COMPLEX.name}'
NAME_FILE_SUFFIX: Final[tuple[str, str]] = ('.npz', '_log.npz')

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] = Path('.') / 'fig' / 'MHD2Dsphere_eigfig'
NAME_FIG: Final[str] \
    = f'MHD2Dsphere_eigfig_NY24_m={M_ORDER}_E={E_ETA}_N={N_T}' \
    if SWITCH_NY24 \
    else f'MHD2Dsphere_eigfig_B{BG_FIELD_B.name}U{BG_FIELD_U.name}' \
    + f'_m={M_ORDER}_E={E_ETA}_R={ROSSBY}_N={N_T}' \
    + f'_{MU_COMPLEX.name}'
NAME_FIG_SUFFIX_1: Final[str] = f'_q={CRITERION_Q}'
NAME_FIG_SUFFIX_2: Final[tuple[str, str, str, str, str, str]] \
    = ('_blk', '_ene', '_psm', '_pse', '_ohm', '_qmode')
NAME_FIG_SUFFIX_3: Final[tuple[str, str]] = ('_R.png', '_I.png')
NAME_FIG_SUFFIX_4: Final[tuple[str, str]] = ('_logR.png', '_logI.png')
NAME_FIG_SUFFIX_5: Final[str] = f'_a={ALPHA_CHOSEN}.png'
FIG_DPI: Final[int] = 600

# The boolean value to switch whether to display the value of the
# magnetic Ekman number or not when E_ETA = 0
SWITCH_DISP_ETA: Final[bool] = False

# The number of processes for multiprocessing
NUM_PROCESS: Final[int] = set_num_process()
# The number of threads for each process
NUM_THREADS: Final[int] = 1

# ================================ #

BG_FIELD: Final[DictBackgroundField] = {
    'B': BG_FIELD_B,
    'U': BG_FIELD_U,
    'MU': MU_COMPLEX,
    'MU_UNUSE_SPECTRAL_DEFORM': MU_COMPLEX_UNUSE_SPECTRAL_DEFORM,
    'NY24': SWITCH_NY24
}

TEX_BG_FIELD: Final[str] \
    = r'$B_{0\phi}=B_0\sin\theta\cos\theta$, $U_{0\phi}=0$' \
    if SWITCH_NY24 \
    else f'{BG_FIELD_B.tex}, {BG_FIELD_U.tex}'

INFO_INPUT: Final[DictFileInfo] = {
    'path_dir': PATH_DIR_INPUT,
    'name_file': NAME_FILE,
    'name_file_suffix': NAME_FILE_SUFFIX
}

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1 if SWITCH_NY24 else N_T + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

# if SWITCH_COLOR == 'ene':
STRETCH_ATAN: Final[float] = 10
COLOR_TICKS: Final[list[float]] = [
    -0.5, -0.2, -0.1, -0.05, -0.02, 0,
    0.02, 0.05, 0.1, 0.2, 0.5]
COLOR_TICKS_ATAN: Final[ArrayFloat] \
    = np.arctan([i_ticks*STRETCH_ATAN for i_ticks in COLOR_TICKS])

TEXT_TITLE: Final[str] \
    = f'Dispersion relation [{TEX_BG_FIELD}] : ' \
    + r'$m=$' + f' {M_ORDER}, ' + r'$R=$' + f' {ROSSBY}' \
    if ((not SWITCH_DISP_ETA) and (E_ETA == 0)) \
    else f'Dispersion relation [{TEX_BG_FIELD}] : ' \
    + r'$m=$' + f' {M_ORDER}, ' + r'$E_\eta=$' + f' {E_ETA}, ' \
    + r'$R=$' + f' {ROSSBY}'
TEXT_XLABEL: Final[str] \
    = r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$'

CBAR_LABEL: Final[str] \
    = 'perturbation kinetic energy' if SWITCH_COLOR == 'ene' else (
        'angular pseudomomentum' if SWITCH_COLOR == 'psm' else (
            'pseudoenergy' if SWITCH_COLOR == 'pse' else (
                'ohmic dissipation' if SWITCH_COLOR == 'ohm' else (
                    r'$\displaystyle{\min_\theta}$'
                    + r'$|(mR\mathcal{U}-\lambda)^2/m^2\alpha^2$'
                    + r'-\mathcal{B}^2|$'
                    if SWITCH_COLOR == 'qmode' else '')
            )
        )
)

NUM_POINT: Final[int] = 10 * N_T
LIN_COLLOCATION_S: Final[ArrayFloat] \
    = np.array([calc_collocation_point(i_l+1, NUM_POINT+2)
                for i_l in range(NUM_POINT)])
LIN_COLLOCATION_MU: Final[ArrayComplex] \
    = np.array([MU_COMPLEX.value(LIN_COLLOCATION_S[i_l])
                for i_l in range(NUM_POINT)])
LIN_BG_FIELD_B: Final[ArrayComplex] \
    = np.array([BG_FIELD_B.value(LIN_COLLOCATION_MU[i_l])
                for i_l in range(NUM_POINT)])
LIN_BG_FIELD_U: Final[ArrayComplex] \
    = np.array([BG_FIELD_U.value(LIN_COLLOCATION_MU[i_l])
                for i_l in range(NUM_POINT)])

MASK_Y1: Final[float] = EIG_IM_LOG_MIN
MASK_Y2: Final[float] = - MASK_Y1


def wrapper_plot_eig(results: DictResult,
                     *,
                     dict_params: DictParams) -> None:
    """Edit the dispersion diagram for the linear-linear plot.

    Parameters
    ----------
    results : DictResult
        The dictionary of results of the eigenvalue problem.
    dict_params : DictParams
        The dictionary of parameters.
    """

    plotter_real: DefaultGridPlotter
    plotter_imag: DefaultGridPlotter
    set_save_fig: set[int]

    plotter_real, plotter_imag, set_save_fig \
        = plot_eig(results, dict_params=dict_params)

    alpha_init: float = dict_params['alpha_init']
    alpha_end: float = dict_params['alpha_end']

    plotter_real.axes[0].set_ylim(EIG_RE_INIT, EIG_RE_END)
    plotter_real.axes[1].set_ylim(EIG_RE_INIT, EIG_RE_END)

    for axis in (plotter_real.axes[0], plotter_real.axes[1],
                 plotter_imag.axes[0], plotter_imag.axes[1]):
        axis.set_xlim(alpha_init, alpha_end)
        axis.set_xlabel(TEXT_XLABEL, fontsize=16)
        axis.tick_params(labelsize=14)

    if set_save_fig & {1, 2}:
        plotter_real.axes[0].set_ylabel(
            r'$\mathrm{Re}(\lambda)=\mathrm{Re}(\omega)/2\Omega_0$',
            fontsize=16)
        plotter_imag.axes[0].set_ylabel(
            r'$\mathrm{Im}(\lambda)=\mathrm{Im}(\omega)/2\Omega_0$',
            fontsize=16)
    else:
        plotter_real.axes[0].set_ylabel(
            r'$\lambda=\omega/2\Omega_0$', fontsize=16)

    plotter_real.axes[0].set_title('Sinuous', fontsize=16)
    plotter_real.axes[1].set_title('Varicose', fontsize=16)
    plotter_imag.axes[0].set_title('Sinuous', fontsize=16)
    plotter_imag.axes[1].set_title('Varicose', fontsize=16)

    plotter_real.fig.suptitle(TEXT_TITLE, fontsize=16)
    plotter_imag.fig.suptitle(TEXT_TITLE, fontsize=16)

    plotter_real.tight_layout()
    plotter_imag.tight_layout()

    axpos: Bbox
    cbar_ax_1: Axes
    cbar_ax_2: Axes

    if SWITCH_COLOR in ('ene', 'ohm', 'psm', 'pse', 'qmode'):
        plotter_real.fig.subplots_adjust(right=0.85)
        axpos = plotter_real.axes[0].get_position()
        cbar_ax_1 = plotter_real.fig.add_axes(
            (0.88, axpos.y0, 0.01, axpos.height))
        if set_save_fig & {1, 2}:
            plotter_imag.fig.subplots_adjust(right=0.85)
            axpos = plotter_imag.axes[0].get_position()
            cbar_ax_2 = plotter_imag.fig.add_axes(
                (0.88, axpos.y0, 0.01, axpos.height))

    cbar: Colorbar

    if SWITCH_COLOR == 'ene':

        cbar = plotter_real.fig.colorbar(
            plotter_real.sc[0], cax=cbar_ax_1, ticks=COLOR_TICKS_ATAN)
        cbar.ax.set_yticklabels(
            [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(label=CBAR_LABEL, size=16)

        if 1 in set_save_fig:
            cbar = plotter_imag.fig.colorbar(
                plotter_imag.sc[0], cax=cbar_ax_2,
                ticks=COLOR_TICKS_ATAN)
            cbar.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(label=CBAR_LABEL, size=16)
        elif 2 in set_save_fig:
            cbar = plotter_imag.fig.colorbar(
                plotter_imag.sc[1], cax=cbar_ax_2,
                ticks=COLOR_TICKS_ATAN)
            cbar.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(label=CBAR_LABEL, size=16)

    elif SWITCH_COLOR in ('psm', 'pse', 'ohm', 'qmode'):

        cbar_extend: str
        if SWITCH_COLOR in ('psm', 'pse'):
            cbar_extend = 'both'
        elif SWITCH_COLOR == 'ohm':
            cbar_extend = 'max'
        elif SWITCH_COLOR == 'qmode':
            cbar_extend = 'min'

        cbar = plotter_real.fig.colorbar(
            plotter_real.sc[0], cax=cbar_ax_1, extend=cbar_extend)
        cbar.set_label(label=CBAR_LABEL, size=16)

        if 1 in set_save_fig:
            cbar = plotter_imag.fig.colorbar(
                plotter_imag.sc[0], cax=cbar_ax_2, extend=cbar_extend)
            cbar.set_label(label=CBAR_LABEL, size=16)
        elif 2 in set_save_fig:
            cbar = plotter_imag.fig.colorbar(
                plotter_imag.sc[1], cax=cbar_ax_2, extend=cbar_extend)
            cbar.set_label(label=CBAR_LABEL, size=16)

    save_plot_eig(plotter_real, plotter_imag, set_save_fig)


def plot_eig(results: DictResult,
             *,
             dict_params: DictParams) -> tuple[DefaultGridPlotter,
                                               DefaultGridPlotter,
                                               set[int]]:
    """Plot the dispersion diagram for the linear-linear plot.

    Parameters
    ----------
    results : DictResult
        The dictionary of results of the eigenvalue problem.
    dict_params : DictParams
        The dictionary of parameters.

    Returns
    -------
    plotter_real : DefaultGridPlotter
        The instance of the DefaultGridPlotter class.
    plotter_imag : DefaultGridPlotter
        The instance of the DefaultGridPlotter class.
    set_save_fig : set[int]
        The set storing the IDs of figures to save.
    """

    lin_alpha: ArrayFloat = results['lin_alpha']
    eig: ArrayComplex = results['eig']
    pke: ArrayFloat = results['phys_qtys']['pke']
    psm: ArrayFloat = results['phys_qtys']['psm']
    pse: ArrayFloat = results['phys_qtys']['pse']
    ohm: ArrayFloat = results['phys_qtys']['ohm']
    sym: ArrayStr = results['phys_qtys']['sym']

    num_alpha: int = dict_params['num_alpha']
    psm_min: float = dict_params['psm_min']
    psm_max: float = dict_params['psm_max']
    pse_min: float = dict_params['pse_min']
    pse_max: float = dict_params['pse_max']
    ohm_max: float = dict_params['ohm_max']

    plotter_real: DefaultGridPlotter \
        = create_plotter(1, 2, figsize=(10, 7))
    plotter_imag: DefaultGridPlotter \
        = create_plotter(1, 2, figsize=(10, 5))

    set_save_fig: set[int] = set()

    cmap_min: float = np.nan
    cmap_max: float = np.nan
    cmap: str
    norm: Normalize | TwoSlopeNorm | LogNorm
    if SWITCH_COLOR == 'ene':
        cmap_min = np.atan(STRETCH_ATAN * (0-0.5))
        cmap_max = np.atan(STRETCH_ATAN * (1-0.5))
        cmap = 'jet'
        norm = Normalize(vmin=cmap_min, vmax=cmap_max)
    elif SWITCH_COLOR == 'psm':
        cmap_min = psm_min
        cmap_max = psm_max
        cmap = 'RdBu_r'
        norm = TwoSlopeNorm(vmin=cmap_min, vcenter=0.0, vmax=cmap_max)
    elif SWITCH_COLOR == 'pse':
        cmap_min = pse_min
        cmap_max = pse_max
        cmap = 'jet'
        norm = Normalize(vmin=cmap_min, vmax=cmap_max)
    elif SWITCH_COLOR == 'ohm':
        cmap_min = 0
        cmap_max = ohm_max
        cmap = 'jet'
        norm = Normalize(vmin=cmap_min, vmax=cmap_max)
    elif SWITCH_COLOR == 'qmode':
        cmap_min = 10**(-6)
        cmap_max = np.max(np.abs(LIN_BG_FIELD_B**2))
        cmap = 'jet'
        norm = LogNorm(vmin=cmap_min, vmax=cmap_max)

    alpha: float
    ones_alpha: ArrayFloat = np.empty(SIZE_MAT, dtype=np.float64)

    dict_eig: dict[str, ArrayComplex]
    scatter_color: ArrayFloat = np.empty(SIZE_MAT, dtype=np.float64)

    for i_alpha in range(num_alpha):
        alpha = lin_alpha[i_alpha]
        ones_alpha = np.full(SIZE_MAT, alpha)

        dict_eig = pickup_eig(
            eig[i_alpha, :], pke[i_alpha, :], sym[i_alpha, :])

        if SWITCH_COLOR == 'blk':

            plotter_real.axes[0].scatter(
                ones_alpha, dict_eig['s'].real, s=0.1, c='black')
            plotter_real.axes[1].scatter(
                ones_alpha, dict_eig['v'].real, s=0.1, c='black')

            if E_ETA == 0:
                if not np.all(np.isnan(dict_eig['s_u'])):
                    plotter_real.axes[0].scatter(
                        ones_alpha, dict_eig['s_u'].real, s=0.2, c='red')
                    plotter_imag.axes[0].scatter(
                        ones_alpha, dict_eig['s_u'].imag, s=0.2, c='red')
                    set_save_fig.add(1)

                if not np.all(np.isnan(dict_eig['v_u'])):
                    plotter_real.axes[1].scatter(
                        ones_alpha, dict_eig['v_u'].real, s=0.2, c='red')
                    plotter_imag.axes[1].scatter(
                        ones_alpha, dict_eig['v_u'].imag, s=0.2, c='red')
                    set_save_fig.add(2)
            else:
                plotter_imag.axes[0].scatter(
                    ones_alpha, dict_eig['s'].imag, s=0.1, c='black')
                plotter_imag.axes[1].scatter(
                    ones_alpha, dict_eig['v'].imag, s=0.1, c='black')
                set_save_fig.update({1, 2})

        elif SWITCH_COLOR in ('ene', 'psm', 'pse', 'ohm', 'qmode'):

            if SWITCH_COLOR == 'ene':
                scatter_color = np.arctan(
                    STRETCH_ATAN * (pke[i_alpha, :]-0.5*np.ones(SIZE_MAT))
                )
            elif SWITCH_COLOR == 'psm':
                scatter_color = psm[i_alpha, :]
            elif SWITCH_COLOR == 'pse':
                scatter_color = pse[i_alpha, :]
            elif SWITCH_COLOR == 'ohm':
                scatter_color = ohm[i_alpha, :]
            elif SWITCH_COLOR == 'qmode':
                scatter_color = np.full(SIZE_MAT, np.nan, dtype=np.float64)
                if alpha != 0:
                    for i_mode in range(SIZE_MAT):
                        scatter_color[i_mode] = np.min(np.abs(
                            (M_ORDER*ROSSBY*LIN_BG_FIELD_U
                             - eig[i_alpha, i_mode])**2
                            / ((M_ORDER*alpha)**2) - (LIN_BG_FIELD_B**2)
                        ))

            if (SWITCH_COLOR == 'ene') and (E_ETA == 0):
                plotter_real.axes[0].scatter(
                    ones_alpha, dict_eig['s_a'].real, s=0.05,
                    c=scatter_color, cmap=cmap, vmin=cmap_min, vmax=cmap_max)
                plotter_real.axes[1].scatter(
                    ones_alpha, dict_eig['v_a'].real, s=0.05,
                    c=scatter_color, cmap=cmap, vmin=cmap_min, vmax=cmap_max)

                plotter_real.sc[0] = plotter_real.axes[0].scatter(
                    ones_alpha, dict_eig['s_na'].real, s=0.2,
                    c=scatter_color, cmap=cmap, vmin=cmap_min, vmax=cmap_max)
                plotter_real.axes[1].scatter(
                    ones_alpha, dict_eig['v_na'].real, s=0.2,
                    c=scatter_color, cmap=cmap, vmin=cmap_min, vmax=cmap_max)

                if not np.all(np.isnan(dict_eig['s_u'])):
                    plotter_imag.sc[0] = plotter_imag.axes[0].scatter(
                        ones_alpha, dict_eig['s_u'].imag, s=0.1,
                        c=scatter_color, cmap=cmap,
                        vmin=cmap_min, vmax=cmap_max)
                    set_save_fig.add(1)

                if not np.all(np.isnan(dict_eig['v_u'])):
                    plotter_imag.sc[1] = plotter_imag.axes[1].scatter(
                        ones_alpha, dict_eig['v_u'].imag, s=0.1,
                        c=scatter_color, cmap=cmap,
                        vmin=cmap_min, vmax=cmap_max)
                    set_save_fig.add(2)
            else:
                plotter_real.sc[0] = plotter_real.axes[0].scatter(
                    ones_alpha, dict_eig['s'].real, s=0.1,
                    c=scatter_color, cmap=cmap, norm=norm)
                plotter_real.axes[1].scatter(
                    ones_alpha, dict_eig['v'].real, s=0.1,
                    c=scatter_color, cmap=cmap, norm=norm)

                plotter_imag.sc[0] = plotter_imag.axes[0].scatter(
                    ones_alpha, dict_eig['s'].imag, s=0.1,
                    c=scatter_color, cmap=cmap, norm=norm)
                plotter_imag.sc[1] = plotter_imag.axes[1].scatter(
                    ones_alpha, dict_eig['v'].imag, s=0.1,
                    c=scatter_color, cmap=cmap, norm=norm)
                set_save_fig.update({1, 2})

    return plotter_real, plotter_imag, set_save_fig


def wrapper_plot_eig_log(results: DictResult,
                         *,
                         dict_params: DictParams) -> None:
    """Edit the dispersion diagram for the log-log plot.

    Parameters
    ----------
    results : DictResult
        The dictionary of results of the eigenvalue problem.
    dict_params : DictParams
        The dictionary of parameters.
    """

    plotter_real: DefaultGridPlotter
    plotter_imag: DefaultGridPlotter
    set_save_fig: set[int]

    plotter_real, plotter_imag, set_save_fig \
        = plot_eig_log(results, dict_params=dict_params)

    alpha_log_init: float = dict_params['alpha_init']
    alpha_log_end: float = dict_params['alpha_end']

    ax_real_all: tuple[Axes, Axes, Axes, Axes] \
        = (plotter_real.axes[0, 0], plotter_real.axes[0, 1],
           plotter_real.axes[1, 0], plotter_real.axes[1, 1])
    ax_imag_all: tuple[Axes, Axes, Axes, Axes] \
        = (plotter_imag.axes[0, 0], plotter_imag.axes[0, 1],
           plotter_imag.axes[1, 0], plotter_imag.axes[1, 1])

    for axis in ax_real_all:
        axis.set_xlim(alpha_log_init, alpha_log_end)
        axis.set_ylim(EIG_RE_LOG_INIT, EIG_RE_LOG_END)
        axis.set_yscale('log')

    for axis in ax_imag_all:
        axis.set_xlim(alpha_log_init, alpha_log_end)
        axis.set_yscale('symlog', linthresh=EIG_IM_LOG_MIN)

    for axis in (ax_real_all + ax_imag_all):
        axis.set_xscale('log')

    plotter_real.axes[1, 0].set_xlabel(TEXT_XLABEL, fontsize=16)
    plotter_real.axes[1, 1].set_xlabel(TEXT_XLABEL, fontsize=16)
    plotter_imag.axes[1, 0].set_xlabel(TEXT_XLABEL, fontsize=16)
    plotter_imag.axes[1, 1].set_xlabel(TEXT_XLABEL, fontsize=16)

    if set_save_fig & {1, 2, 3, 4}:
        plotter_real.axes[0, 0].set_ylabel(
            r'$|\mathrm{Re}(\lambda)|=|\mathrm{Re}(\omega)/2\Omega_0|$',
            fontsize=16)
        plotter_real.axes[1, 0].set_ylabel(
            r'$|\mathrm{Re}(\lambda)|=|\mathrm{Re}(\omega)/2\Omega_0|$',
            fontsize=16)
        plotter_imag.axes[0, 0].set_ylabel(
            r'$\mathrm{Im}(\lambda)=\mathrm{Im}(\omega)/2\Omega_0$',
            fontsize=16)
        plotter_imag.axes[1, 0].set_ylabel(
            r'$\mathrm{Im}(\lambda)=\mathrm{Im}(\omega)/2\Omega_0$',
            fontsize=16)

        plotter_real.axes[0, 0].set_title(
            r'Sinuous, Retrograde ($\mathrm{Re}(\lambda)<0$)',
            fontsize=16)
        plotter_real.axes[0, 1].set_title(
            r'Sinuous, Prograde ($\mathrm{Re}(\lambda)>0$)',
            fontsize=16)
        plotter_real.axes[1, 0].set_title(
            r'Varicose, Retrograde ($\mathrm{Re}(\lambda)<0$)',
            fontsize=16)
        plotter_real.axes[1, 1].set_title(
            r'Varicose, Prograde ($\mathrm{Re}(\lambda)>0$)',
            fontsize=16)
        plotter_imag.axes[0, 0].set_title(
            r'Sinuous, Retrograde ($\mathrm{Re}(\lambda)<0$)',
            fontsize=16)
        plotter_imag.axes[0, 1].set_title(
            r'Sinuous, Prograde ($\mathrm{Re}(\lambda)>0$)',
            fontsize=16)
        plotter_imag.axes[1, 0].set_title(
            r'Varicose, Retrograde ($\mathrm{Re}(\lambda)<0$)',
            fontsize=16)
        plotter_imag.axes[1, 1].set_title(
            r'Varicose, Prograde ($\mathrm{Re}(\lambda)>0$)',
            fontsize=16)
    else:
        plotter_real.axes[0, 0].set_ylabel(
            r'$|\lambda|=|\omega/2\Omega_0|$', fontsize=16)
        plotter_real.axes[1, 0].set_ylabel(
            r'$|\lambda|=|\omega/2\Omega_0|$', fontsize=16)

        plotter_real.axes[0, 0].set_title(
            r'Sinuous, Retrograde ($\lambda<0$)', fontsize=16)
        plotter_real.axes[0, 1].set_title(
            r'Sinuous, Prograde ($\lambda>0$)', fontsize=16)
        plotter_real.axes[1, 0].set_title(
            r'Varicose, Retrograde ($\lambda<0$)', fontsize=16)
        plotter_real.axes[1, 1].set_title(
            r'Varicose, Prograde ($\lambda>0$)', fontsize=16)

    for axis in (ax_real_all + ax_imag_all):
        axis.tick_params(labelsize=12)

    plotter_real.fig.suptitle(TEXT_TITLE, fontsize=16)
    plotter_imag.fig.suptitle(TEXT_TITLE, fontsize=16)

    plotter_real.tight_layout()
    plotter_imag.tight_layout()

    axpos1: Bbox
    axpos2: Bbox
    cbar_ax_1: Axes
    cbar_ax_2: Axes

    if SWITCH_COLOR in ('ene', 'ohm', 'psm', 'pse', 'qmode'):
        plotter_real.fig.subplots_adjust(right=0.85)
        axpos1 = plotter_real.axes[0, 0].get_position()
        cbar_ax_1 = plotter_real.fig.add_axes(
            (0.88, axpos1.y0, 0.01, axpos1.height))
        if set_save_fig & {1, 2, 3, 4}:
            plotter_imag.fig.subplots_adjust(right=0.85)
            axpos2 = plotter_imag.axes[0, 0].get_position()
            cbar_ax_2 = plotter_imag.fig.add_axes(
                (0.88, axpos2.y0, 0.01, axpos2.height))

    cbar1: Colorbar
    cbar2: Colorbar

    if SWITCH_COLOR == 'ene':

        cbar1 = plotter_real.fig.colorbar(
            plotter_real.sc[0, 0], cax=cbar_ax_1, ticks=COLOR_TICKS_ATAN)
        cbar1.ax.set_yticklabels(
            [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
        cbar1.ax.tick_params(labelsize=14)
        cbar1.set_label(label=CBAR_LABEL, size=16)

        if 1 in set_save_fig:
            cbar2 = plotter_imag.fig.colorbar(
                plotter_imag.sc[0, 0], cax=cbar_ax_2,
                ticks=COLOR_TICKS_ATAN)
            cbar2.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 2 in set_save_fig:
            cbar2 = plotter_imag.fig.colorbar(
                plotter_imag.sc[0, 1], cax=cbar_ax_2,
                ticks=COLOR_TICKS_ATAN)
            cbar2.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 3 in set_save_fig:
            cbar2 = plotter_imag.fig.colorbar(
                plotter_imag.sc[1, 0], cax=cbar_ax_2,
                ticks=COLOR_TICKS_ATAN)
            cbar2.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 4 in set_save_fig:
            cbar2 = plotter_imag.fig.colorbar(
                plotter_imag.sc[1, 1], cax=cbar_ax_2,
                ticks=COLOR_TICKS_ATAN)
            cbar2.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label=CBAR_LABEL, size=16)

    elif SWITCH_COLOR in ('ohm', 'psm', 'pse', 'qmode'):

        cbar_extend: str
        if SWITCH_COLOR in ('psm', 'pse'):
            cbar_extend = 'both'
        elif SWITCH_COLOR == 'ohm':
            cbar_extend = 'max'
        elif SWITCH_COLOR == 'qmode':
            cbar_extend = 'min'

        cbar1 = plotter_real.fig.colorbar(
            plotter_real.sc[0, 0], cax=cbar_ax_1, extend=cbar_extend)
        cbar1.set_label(label=CBAR_LABEL, size=16)
        if 1 in set_save_fig:
            cbar2 = plotter_imag.fig.colorbar(
                plotter_imag.sc[0, 0], cax=cbar_ax_2, extend=cbar_extend)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 2 in set_save_fig:
            cbar2 = plotter_imag.fig.colorbar(
                plotter_imag.sc[0, 1], cax=cbar_ax_2, extend=cbar_extend)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 3 in set_save_fig:
            cbar2 = plotter_imag.fig.colorbar(
                plotter_imag.sc[1, 0], cax=cbar_ax_2, extend=cbar_extend)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 4 in set_save_fig:
            cbar2 = plotter_imag.fig.colorbar(
                plotter_imag.sc[1, 1], cax=cbar_ax_2, extend=cbar_extend)
            cbar2.set_label(label=CBAR_LABEL, size=16)

    # For Fig. 6 in Nakashima and Yoshida (2024)
    # ax1[0, 1].scatter(0.013, 0.00012, s=50, c='white', marker='*',
    #                   linewidth=0.5, edgecolors="black")
    # ax1[0, 1].scatter(0.013, 0.00025, s=50, c='white', marker='*',
    #                   linewidth=0.5, edgecolors="black")

    save_plot_eig(plotter_real, plotter_imag, set_save_fig,
                  switch_log=True)


def plot_eig_log(results: DictResult,
                 *,
                 dict_params: DictParams) -> tuple[DefaultGridPlotter,
                                                   DefaultGridPlotter,
                                                   set[int]]:
    """Plot the dispersion diagram for the log-log plot.

    Parameters
    ----------
    results : DictResult
        The dictionary of results of the eigenvalue problem.
    dict_params : DictParams
        The dictionary of parameters.

    Returns
    -------
    plotter_real : DefaultGridPlotter
        The instance of the DefaultGridPlotter class.
    plotter_imag : DefaultGridPlotter
        The instance of the DefaultGridPlotter class.
    set_save_fig : set[int]
        The set storing the IDs of figures to save.
    """

    lin_alpha: ArrayFloat = results['lin_alpha']
    eig: ArrayComplex = results['eig']
    pke: ArrayFloat = results['phys_qtys']['pke']
    psm: ArrayFloat = results['phys_qtys']['psm']
    pse: ArrayFloat = results['phys_qtys']['pse']
    ohm: ArrayFloat = results['phys_qtys']['ohm']
    sym: ArrayStr = results['phys_qtys']['sym']

    num_alpha_log: int = dict_params['num_alpha']
    psm_min: float = dict_params['psm_min']
    psm_max: float = dict_params['psm_max']
    pse_min: float = dict_params['pse_min']
    pse_max: float = dict_params['pse_max']
    ohm_log_max: float = dict_params['ohm_max']

    plotter_real: DefaultGridPlotter \
        = create_plotter(2, 2, figsize=(10, 10))
    plotter_imag: DefaultGridPlotter \
        = create_plotter(2, 2, figsize=(10, 10))

    set_save_fig: set[int] = set()

    cmap_min: float = np.nan
    cmap_max: float = np.nan
    norm: Normalize | TwoSlopeNorm | LogNorm
    if SWITCH_COLOR == 'ene':
        cmap_min = np.atan(STRETCH_ATAN * (0-0.5))
        cmap_max = np.atan(STRETCH_ATAN * (1-0.5))
        cmap = 'jet'
        norm = Normalize(vmin=cmap_min, vmax=cmap_max)
    elif SWITCH_COLOR == 'psm':
        cmap_min = psm_min
        cmap_max = psm_max
        cmap = 'RdBu_r'
        norm = TwoSlopeNorm(vmin=cmap_min, vcenter=0.0, vmax=cmap_max)
    elif SWITCH_COLOR == 'pse':
        cmap_min = pse_min
        cmap_max = pse_max
        cmap = 'jet'
        norm = Normalize(vmin=cmap_min, vmax=cmap_max)
    elif SWITCH_COLOR == 'ohm':
        cmap_min = 0
        cmap_max = ohm_log_max
        cmap = 'jet'
        norm = Normalize(vmin=cmap_min, vmax=cmap_max)
    elif SWITCH_COLOR == 'qmode':
        cmap_min = 10**(-6)
        cmap_max = np.max(np.abs(LIN_BG_FIELD_B**2))
        cmap = 'jet'
        norm = LogNorm(vmin=cmap_min, vmax=cmap_max)

    alpha: float
    ones_alpha: ArrayFloat = np.empty(SIZE_MAT, dtype=np.float64)

    dict_eig: dict[str, ArrayComplex]
    scatter_color: ArrayFloat = np.empty(SIZE_MAT, dtype=np.float64)

    for i_alpha in range(num_alpha_log):
        alpha = lin_alpha[i_alpha]
        ones_alpha = np.full(SIZE_MAT, alpha)

        dict_eig = pickup_eig(
            eig[i_alpha, :], pke[i_alpha, :], sym[i_alpha, :])

        dict_eig['sr'] = -np.conjugate(dict_eig['sr'])
        dict_eig['vr'] = -np.conjugate(dict_eig['vr'])
        dict_eig['sr_u'] = -np.conjugate(dict_eig['sr_u'])
        dict_eig['vr_u'] = -np.conjugate(dict_eig['vr_u'])
        dict_eig['sr_a'] = -np.conjugate(dict_eig['sr_a'])
        dict_eig['vr_a'] = -np.conjugate(dict_eig['vr_a'])
        dict_eig['sr_na'] = -np.conjugate(dict_eig['sr_na'])
        dict_eig['vr_na'] = -np.conjugate(dict_eig['vr_na'])

        if SWITCH_COLOR == 'blk':

            plotter_real.axes[0, 0].scatter(
                ones_alpha, dict_eig['sr'].real, s=0.1, c='black')
            plotter_real.axes[0, 1].scatter(
                ones_alpha, dict_eig['sp'].real, s=0.1, c='black')
            plotter_real.axes[1, 0].scatter(
                ones_alpha, dict_eig['vr'].real, s=0.1, c='black')
            plotter_real.axes[1, 1].scatter(
                ones_alpha, dict_eig['vp'].real, s=0.1, c='black')

            if E_ETA == 0:
                if not np.all(np.isnan(dict_eig['sr_u'])):
                    plotter_real.axes[0, 0].scatter(
                        ones_alpha, dict_eig['sr_u'].real,
                        s=0.2, c='red')
                    plotter_imag.axes[0, 0].scatter(
                        ones_alpha, dict_eig['sr_u'].imag,
                        s=0.2, c='red')
                    set_save_fig.add(1)

                if not np.all(np.isnan(dict_eig['sp_u'])):
                    plotter_real.axes[0, 1].scatter(
                        ones_alpha, dict_eig['sp_u'].real,
                        s=0.2, c='red')
                    plotter_imag.axes[0, 1].scatter(
                        ones_alpha, dict_eig['sp_u'].imag,
                        s=0.2, c='red')
                    set_save_fig.add(2)

                if not np.all(np.isnan(dict_eig['vr_u'])):
                    plotter_real.axes[1, 0].scatter(
                        ones_alpha, dict_eig['vr_u'].real,
                        s=0.2, c='red')
                    plotter_imag.axes[1, 0].scatter(
                        ones_alpha, dict_eig['vr_u'].imag,
                        s=0.2, c='red')
                    set_save_fig.add(3)

                if not np.all(np.isnan(dict_eig['vp_u'])):
                    plotter_real.axes[1, 1].scatter(
                        ones_alpha, dict_eig['vp_u'].real,
                        s=0.2, c='red')
                    plotter_imag.axes[1, 1].scatter(
                        ones_alpha, dict_eig['vp_u'].imag,
                        s=0.2, c='red')
                    set_save_fig.add(4)
            else:
                plotter_imag.axes[0, 0].scatter(
                    ones_alpha, dict_eig['sr'].imag, s=0.1, c='black')
                plotter_imag.axes[0, 1].scatter(
                    ones_alpha, dict_eig['sp'].imag, s=0.1, c='black')
                plotter_imag.axes[1, 0].scatter(
                    ones_alpha, dict_eig['vr'].imag, s=0.1, c='black')
                plotter_imag.axes[1, 1].scatter(
                    ones_alpha, dict_eig['vp'].imag, s=0.1, c='black')
                set_save_fig.update({1, 2, 3, 4})

        elif SWITCH_COLOR in ('ene', 'ohm', 'psm', 'pse', 'qmode'):

            if SWITCH_COLOR == 'ene':
                scatter_color = np.arctan(
                    STRETCH_ATAN
                    * (pke[i_alpha, :]-0.5*np.ones(SIZE_MAT))
                )
            elif SWITCH_COLOR == 'ohm':
                scatter_color = ohm[i_alpha, :]
            elif SWITCH_COLOR == 'psm':
                scatter_color = psm[i_alpha, :]
            elif SWITCH_COLOR == 'pse':
                scatter_color = pse[i_alpha, :]
            elif SWITCH_COLOR == 'qmode':
                scatter_color = np.full(SIZE_MAT, np.nan, dtype=np.float64)
                if alpha != 0:
                    for i_mode in range(SIZE_MAT):
                        scatter_color[i_mode] = np.min(np.abs(
                            (M_ORDER*ROSSBY*LIN_BG_FIELD_U
                             - eig[i_alpha, i_mode])**2
                            / ((M_ORDER*alpha)**2) - (LIN_BG_FIELD_B**2)
                        ))

            if (SWITCH_COLOR == 'ene') and (E_ETA == 0):
                plotter_real.axes[0, 0].scatter(
                    ones_alpha, dict_eig['sr_a'].real, s=0.05,
                    c=scatter_color, cmap=cmap,
                    vmin=cmap_min, vmax=cmap_max)
                plotter_real.axes[0, 1].scatter(
                    ones_alpha, dict_eig['sp_a'].real, s=0.05,
                    c=scatter_color, cmap=cmap,
                    vmin=cmap_min, vmax=cmap_max)
                plotter_real.axes[1, 0].scatter(
                    ones_alpha, dict_eig['vr_a'].real, s=0.05,
                    c=scatter_color, cmap=cmap,
                    vmin=cmap_min, vmax=cmap_max)
                plotter_real.axes[1, 1].scatter(
                    ones_alpha, dict_eig['vp_a'].real, s=0.05,
                    c=scatter_color, cmap=cmap,
                    vmin=cmap_min, vmax=cmap_max)

                plotter_real.sc[0, 0] = plotter_real.axes[0, 0].scatter(
                    ones_alpha, dict_eig['sr_na'].real, s=0.2,
                    c=scatter_color, cmap=cmap,
                    vmin=cmap_min, vmax=cmap_max)
                plotter_real.axes[0, 1].scatter(
                    ones_alpha, dict_eig['sp_na'].real, s=0.2,
                    c=scatter_color, cmap=cmap,
                    vmin=cmap_min, vmax=cmap_max)
                plotter_real.axes[1, 0].scatter(
                    ones_alpha, dict_eig['vr_na'].real, s=0.2,
                    c=scatter_color, cmap=cmap,
                    vmin=cmap_min, vmax=cmap_max)
                plotter_real.axes[1, 1].scatter(
                    ones_alpha, dict_eig['vp_na'].real, s=0.2,
                    c=scatter_color, cmap=cmap,
                    vmin=cmap_min, vmax=cmap_max)

                if not np.all(np.isnan(dict_eig['sr_u'])):
                    plotter_imag.sc[0, 0] \
                        = plotter_imag.axes[0, 0].scatter(
                        ones_alpha, dict_eig['sr_u'].imag, s=0.1,
                        c=scatter_color, cmap=cmap,
                        vmin=cmap_min, vmax=cmap_max)
                    set_save_fig.add(1)

                if not np.all(np.isnan(dict_eig['sp_u'])):
                    plotter_imag.sc[0, 1] \
                        = plotter_imag.axes[0, 1].scatter(
                        ones_alpha, dict_eig['sp_u'].imag, s=0.1,
                        c=scatter_color, cmap=cmap,
                        vmin=cmap_min, vmax=cmap_max)
                    set_save_fig.add(2)

                if not np.all(np.isnan(dict_eig['vr_u'])):
                    plotter_imag.sc[1, 0] \
                        = plotter_imag.axes[1, 0].scatter(
                        ones_alpha, dict_eig['vr_u'].imag, s=0.1,
                        c=scatter_color, cmap=cmap,
                        vmin=cmap_min, vmax=cmap_max)
                    set_save_fig.add(3)

                if not np.all(np.isnan(dict_eig['vp_u'])):
                    plotter_imag.sc[1, 1] \
                        = plotter_imag.axes[1, 1].scatter(
                        ones_alpha, dict_eig['vp_u'].imag, s=0.1,
                        c=scatter_color, cmap=cmap,
                        vmin=cmap_min, vmax=cmap_max)
                    set_save_fig.add(4)
            else:
                plotter_real.sc[0, 0] = plotter_real.axes[0, 0].scatter(
                    ones_alpha, dict_eig['sr'].real, s=0.1,
                    c=scatter_color, cmap=cmap, norm=norm)
                plotter_real.axes[0, 1].scatter(
                    ones_alpha, dict_eig['sp'].real, s=0.1,
                    c=scatter_color, cmap=cmap, norm=norm)
                plotter_real.axes[1, 0].scatter(
                    ones_alpha, dict_eig['vr'].real, s=0.1,
                    c=scatter_color, cmap=cmap, norm=norm)
                plotter_real.axes[1, 1].scatter(
                    ones_alpha, dict_eig['vp'].real, s=0.1,
                    c=scatter_color, cmap=cmap, norm=norm)

                if not np.all(np.isnan(dict_eig['sr_u'])):
                    plotter_imag.sc[0, 0] = plotter_imag.axes[0, 0].scatter(
                        ones_alpha, dict_eig['sr'].imag, s=0.1,
                        c=scatter_color, cmap=cmap, norm=norm)
                    set_save_fig.add(1)
                if not np.all(np.isnan(dict_eig['sp_u'])):
                    plotter_imag.sc[0, 1] = plotter_imag.axes[0, 1].scatter(
                        ones_alpha, dict_eig['sp'].imag, s=0.1,
                        c=scatter_color, cmap=cmap, norm=norm)
                    set_save_fig.add(2)
                if not np.all(np.isnan(dict_eig['vr_u'])):
                    plotter_imag.sc[1, 0] = plotter_imag.axes[1, 0].scatter(
                        ones_alpha, dict_eig['vr'].imag, s=0.1,
                        c=scatter_color, cmap=cmap, norm=norm)
                    set_save_fig.add(3)
                if not np.all(np.isnan(dict_eig['vp_u'])):
                    plotter_imag.sc[1, 1] = plotter_imag.axes[1, 1].scatter(
                        ones_alpha, dict_eig['vp'].imag, s=0.1,
                        c=scatter_color, cmap=cmap, norm=norm)
                    set_save_fig.add(4)

    mask_x: ArrayFloat = lin_alpha
    plotter_imag.axes[0, 0].fill_between(
        mask_x, MASK_Y1, MASK_Y2, facecolor='gray')
    plotter_imag.axes[0, 1].fill_between(
        mask_x, MASK_Y1, MASK_Y2, facecolor='gray')
    plotter_imag.axes[1, 0].fill_between(
        mask_x, MASK_Y1, MASK_Y2, facecolor='gray')
    plotter_imag.axes[1, 1].fill_between(
        mask_x, MASK_Y1, MASK_Y2, facecolor='gray')

    return plotter_real, plotter_imag, set_save_fig


def save_plot_eig(plotter_real: DefaultGridPlotter,
                  plotter_imag: DefaultGridPlotter,
                  set_save_fig: set[int],
                  *,
                  switch_log: bool = False) -> None:
    """Save the dispersion diagram.

    Parameters
    ----------
    plotter_real : DefaultGridPlotter
        The instance of the DefaultGridPlotter class.
    plotter_imag : DefaultGridPlotter
        The instance of the DefaultGridPlotter class.
    set_save_fig : set[int]
        The set storing the IDs of figures to save.
    switch_log : bool, optional, default False
        The boolean value for the dispersion diagram of the log-log plot.
    """

    name_fig: str
    if (E_ETA != 0) and (CRITERION_Q > 0):
        name_fig = NAME_FIG + NAME_FIG_SUFFIX_1
    else:
        name_fig = NAME_FIG

    if SWITCH_COLOR == 'blk':
        name_fig += NAME_FIG_SUFFIX_2[0]
    elif SWITCH_COLOR == 'ene':
        name_fig += NAME_FIG_SUFFIX_2[1]
    elif SWITCH_COLOR == 'psm':
        name_fig += NAME_FIG_SUFFIX_2[2]
    elif SWITCH_COLOR == 'pse':
        name_fig += NAME_FIG_SUFFIX_2[3]
    elif SWITCH_COLOR == 'ohm':
        name_fig += NAME_FIG_SUFFIX_2[4]
    elif SWITCH_COLOR == 'qmode':
        name_fig += NAME_FIG_SUFFIX_2[5]

    list_name_fig: list[str]
    if not switch_log:
        list_name_fig \
            = [name_fig + suffix for suffix in NAME_FIG_SUFFIX_3]
    else:
        list_name_fig \
            = [name_fig + suffix for suffix in NAME_FIG_SUFFIX_4]

    plotter_real.save(PATH_DIR_FIG, list_name_fig[0], FIG_DPI,
                      switch_tight_layout=False)
    if (not switch_log) and (set_save_fig & {1, 2}):
        plotter_imag.save(PATH_DIR_FIG, list_name_fig[1], FIG_DPI,
                          switch_tight_layout=False)
    if switch_log and (set_save_fig & {1, 2, 3, 4}):
        plotter_imag.save(PATH_DIR_FIG, list_name_fig[1], FIG_DPI,
                          switch_tight_layout=False)


def plot_eig_for_an_alpha(results: DictResult) -> None:
    """Plot the dispersion diagram for a chosen alpha.

    Parameters
    ----------
    results : DictResult
        The dictionary of results of the eigenvalue problem.
    """

    lin_alpha: ArrayFloat = results['lin_alpha']
    i_alpha: int = int(np.argmin(np.abs(lin_alpha - ALPHA_CHOSEN)))
    alpha: float = lin_alpha[i_alpha]

    eig: ArrayComplex = results['eig'][i_alpha, :]
    eig_center: complex = (np.nanmax(eig.real)+np.nanmin(eig.real)) / 2 \
        + 1j * (np.nanmax(eig.imag)+np.nanmin(eig.imag)) / 2
    half_width: float = max(
        np.nanmax(eig.real) - np.nanmin(eig.real),
        np.nanmax(eig.imag) - np.nanmin(eig.imag)
    ) / 2
    eig_range: tuple[float, float, float, float] = (
        eig_center.real - 1.1*half_width, eig_center.real + 1.1*half_width,
        eig_center.imag - 1.1*half_width, eig_center.imag + 1.1*half_width)

    lin_re: ArrayFloat = np.linspace(eig_range[0], eig_range[1], NUM_EIG_GRID)
    lin_im: ArrayFloat = np.linspace(eig_range[2], eig_range[3], NUM_EIG_GRID)
    grid_re, grid_im = np.meshgrid(lin_re, lin_im)

    pseudospectrum: ArrayFloat = calc_pseudospectrum(alpha, eig_range)

    plotter: DefaultPlotter = create_plotter(1, 1, figsize=(7, 5))

    contour: QuadContourSet = plotter.axes.contourf(
        grid_re, grid_im, pseudospectrum, cmap='Blues_r', norm=LogNorm())

    cbar: Colorbar = plotter.fig.colorbar(contour, ax=plotter.axes)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(
        label=r'$\|(\lambda \mathsf{I}-\mathsf{A})^{-1}\|$', size=16)

    plotter.axes.scatter(eig.real, eig.imag, s=2, c='black')

    plotter.axes.set_xlim(eig_range[0], eig_range[1])
    plotter.axes.set_ylim(eig_range[2], eig_range[3])

    plotter.axes.set_aspect('equal')

    plotter.axes.set_xlabel(
        r'$\mathrm{Re}(\lambda)=\mathrm{Re}(\omega)/2\Omega_0$',
        fontsize=16)
    plotter.axes.set_ylabel(
        r'$\mathrm{Im}(\lambda)=\mathrm{Im}(\omega)/2\Omega_0$',
        fontsize=16)
    plotter.axes.set_title(
        TEXT_TITLE + '\n'
        + r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|=$'
        + f' {alpha}', fontsize=16)

    plotter.axes.tick_params(labelsize=14)

    plotter.save(PATH_DIR_FIG, NAME_FIG + NAME_FIG_SUFFIX_5, FIG_DPI)


def calc_pseudospectrum(
        alpha: float,
        eig_range: tuple[float, float, float, float]) -> ArrayFloat:
    """Calculate the epsilon-pseudospectrum for a chosen alpha.

    Parameters
    ----------
    alpha : float
        The Lehnert number.
    eig_range : tuple[float, float, float, float]
        The range of the eigenvalues in the complex plane.

    Returns
    -------
    pseudospectrum : ArrayFloat
        The minimum singular values on the complex grid.
    """

    lin_re: ArrayFloat = np.linspace(eig_range[0], eig_range[1], NUM_EIG_GRID)
    lin_im: ArrayFloat = np.linspace(eig_range[2], eig_range[3], NUM_EIG_GRID)

    submatrices: tuple[ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex] \
        = create_submat(M_ORDER, E_ETA, ROSSBY, SIZE_SUBMAT,
                        background_field=BG_FIELD)

    mat: ArrayFloat | ArrayComplex = create_mat(
        M_ORDER, alpha, E_ETA, submatrices, background_field=BG_FIELD)

    identity: ArrayFloat = np.identity(mat.shape[0], dtype=np.float64)

    shared_memories: tuple[SharedMemory, ...]
    shared_info: SharedInfo
    shared_memories, shared_info = create_shared_arrays(lin_re, mat, identity)

    try:
        args_list: list[tuple[float, SharedInfo]] = [
            (eig_im, shared_info) for eig_im in lin_im
        ]

        pseudospectrum: ArrayFloat = np.empty(
            (NUM_EIG_GRID, NUM_EIG_GRID), dtype=np.float64)

        progress_bar: ProgressBar \
            = create_function_name_progress_bar(NUM_EIG_GRID)
        progress_bar.start()
        with multiprocessing.Pool(processes=NUM_PROCESS,
                                  initializer=set_num_threads,
                                  initargs=(NUM_THREADS,)) as pool:
            for i_im, result in enumerate(pool.imap(worker, args_list)):

                pseudospectrum[i_im, :] = result
                progress_bar.update(i_im, NUM_PROCESS)

    finally:
        detach_shared_arrays(*shared_memories, unlink=True)

    return pseudospectrum


def worker(args: tuple[float, SharedInfo]) -> ArrayFloat:
    """Set the task for multiprocessing

    Parameters
    ----------
    args : tuple[float, SharedInfo]
        The arguments for the task

    Returns
    -------
    result : ArrayFloat
        The result of the task
    """

    eig_im: float
    shared_info: SharedInfo
    eig_im, shared_info = args

    shared_memories_tmp: tuple[SharedMemory, ...]
    shared_arrays_tmp: tuple[ArrayFloat | ArrayComplex, ...]
    shared_memories_tmp, shared_arrays_tmp = attach_shared_arrays(shared_info)
    shared_memories = cast(tuple[SharedMemory,
                                 SharedMemory,
                                 SharedMemory], shared_memories_tmp)
    shared_arrays = cast(tuple[ArrayFloat,
                               ArrayFloat | ArrayComplex,
                               ArrayFloat], shared_arrays_tmp)
    lin_re, mat, identity = shared_arrays

    pseudospectrum: ArrayFloat = np.empty(NUM_EIG_GRID, dtype=np.float64)

    for i_re, real in enumerate(lin_re):
        singular_values: ArrayFloat \
            = cast(ArrayFloat,
                   svdvals((real + 1j * eig_im) * identity - mat,
                           overwrite_a=True, check_finite=False))

        pseudospectrum[i_re] = 1 / singular_values[-1]

    detach_shared_arrays(*shared_memories)

    return pseudospectrum


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    logger: DefaultLogger = DefaultLogger(__name__)

    if True not in SWITCH_PLOT:
        logger.info('No plotted figures')
        sys.exit(0)

    if SWITCH_COLOR not in ('blk', 'ene', 'ohm', 'psm', 'pse', 'qmode'):
        logger.error('Invalid value for \'SWITCH_COLOR\'')

    if (SWITCH_COLOR == 'blk') and (
            MU_COMPLEX.use_spectral_deform or (E_ETA != 0)):
        logger.warning('Meaningless figures are plotted')
        sys.exit(0)

    if (SWITCH_COLOR == 'ohm') and (E_ETA == 0):
        logger.warning('Meaningless figures are plotted')
        sys.exit(0)

    if (SWITCH_COLOR == 'qmode') \
            and (not MU_COMPLEX.use_spectral_deform):
        logger.warning('Meaningless figures are plotted')
        sys.exit(0)

    if SWITCH_NY24:
        logger.show_params(f'{SWITCH_NY24=}',
                           f'{M_ORDER=}',
                           f'{E_ETA=}',
                           f'{ROSSBY=}',
                           f'{N_T=}')
    else:
        logger.show_params(f'{BG_FIELD_B.name=}',
                           f'{BG_FIELD_U.name=}',
                           f'{MU_COMPLEX.name=}',
                           f'{M_ORDER=}',
                           f'{E_ETA=}',
                           f'{ROSSBY=}',
                           f'{N_T=}')

    data: DictResult | None
    data_log: DictResult | None
    data, data_log = wrapper_load_results(SWITCH_PLOT, info_load=INFO_INPUT)

    params: DictParams
    if SWITCH_PLOT[0] and (data is not None):

        if (E_ETA != 0) and (CRITERION_Q > 0):
            data = screening_eig_q(
                data, criterion_q=CRITERION_Q)

        params = pickup_param(data)

        wrapper_plot_eig(data, dict_params=params)

    if SWITCH_PLOT[1] and (data_log is not None):

        if (E_ETA != 0) and (CRITERION_Q > 0):
            data_log = screening_eig_q(
                data_log, criterion_q=CRITERION_Q)

        params_log = pickup_param(data_log)

        wrapper_plot_eig_log(data_log, dict_params=params_log)

    if SWITCH_PLOT[2] and (data is not None):

        if (E_ETA != 0) and (CRITERION_Q > 0):
            data = screening_eig_q(
                data, criterion_q=CRITERION_Q)

        plot_eig_for_an_alpha(data)

    timer.end()

    plt.show()
