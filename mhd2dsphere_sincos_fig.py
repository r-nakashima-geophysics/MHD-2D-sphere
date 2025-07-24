"""A code for 2D MHD waves on a rotating sphere under the non-Malkus
field B_phi = B_0 sin(theta) cos(theta)

Plots 1-4 figures (linear-linear and log-log / black, energy
partitioning, and ohmic dissipation) concerning the dispersion relation
for 2D MHD waves on a rotating sphere under the non-Malkus field
B_phi = B_0 sin(theta) cos(theta).

Parameters
-----
M_ORDER : int
    The zonal wavenumber (order)

Raises
-----
No plotted figures
    If all of the boolean values to switch whether to plot figures are
    False.
Invalid value for 'SWITCH_COLOR'
    If 'SWITCH_COLOR' is not either 'blk', 'ene', or 'ohm'.
Meaningless figures are plotted
    If figures of the ohmic dissipation are plotted in the ideal MHD
    case.

Notes
-----
Parameters other than command line arguments are described below. You
must run mhd2dsphere_sincos.py with the same parameters before
executing this code.

References
-----
[1] Nakashima & Yoshida (submitted)

Examples
-----
In the below example, M_ORDER will be set to the default value.
    python3 mhd2dsphere_sincos_fig.py
In the below example, M_ORDER will be set to 2.
    python3 mhd2dsphere_sincos_fig.py 2

"""

import logging
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from package.input_arg import input_int
from package.load_data import wrapper_load_results
from package.processing_results import (pickup_eig, pickup_param,
                                        screening_eig_q)

# ========== Parameters ==========

# Boolean values to switch whether to plot figures
# 0: dispersion relation (linear-linear)
# 1: dispersion relation (log-log)
SWITCH_PLOT: Final[tuple[bool, bool]] = (True, True)

# The rule of the coloring of plots
# black ('blk')
# energy partitioning ('ene')
# ohmic dissipation ('ohm')
SWITCH_COLOR: Final[str] = 'ene'

# The boolean value to switch whether to display the value of the
# magnetic Ekman number when E_ETA = 0
SWITCH_DISP_ETA: Final[bool] = False

# The zonal wavenumber (order)
M_ORDER: Final[int] = input_int(1)

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The truncation degree
N_T: Final[int] = 2000

# A criterion for plotting eigenvalues based on the Q value
CRITERION_Q: Final[float] = 0

# The range of eigenvalues
# linear, real part
EIG_RE_INIT: Final[float] = -2
EIG_RE_END: Final[float] = 2
# log, real part
EIG_RE_LOG_INIT: Final[float] = -6
EIG_RE_LOG_END: Final[float] = 2
# log, imaginary part
EIG_IM_LOG_MIN: Final[float] = -6

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_sincos_fig'
NAME_FIG: Final[str] \
    = f'MHD2Dsphere_sincos_fig_m{M_ORDER}E{E_ETA}N{N_T}'
NAME_FIG_SUFFIX_1: Final[str] = f'q{CRITERION_Q}'
NAME_FIG_SUFFIX_2: Final[tuple[str, str, str]] \
    = ('_eig', '_eigene', '_eigohm')
NAME_FIG_SUFFIX_3: Final[tuple[str, str]] = ('R.png', 'I.png')
NAME_FIG_SUFFIX_4: Final[tuple[str, str]] = ('logR.png', 'logI.png')
FIG_DPI: Final[int] = 600

# ================================

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

# if SWITCH_COLOR == 'ene':
STRETCH_ATAN: Final[float] = 10
COLOR_TICKS: Final[list[float]] = [
    -0.5, -0.2, -0.1, -0.05, -0.02, 0,
    0.02, 0.05, 0.1, 0.2, 0.5]
COLOR_TICKS_ATAN: Final[np.ndarray] \
    = np.arctan([i_ticks*STRETCH_ATAN for i_ticks in COLOR_TICKS])

CBAR_LABEL: str = str()
if SWITCH_COLOR == 'ene':
    CBAR_LABEL = 'mean kinetic energy'
elif SWITCH_COLOR == 'ohm':
    CBAR_LABEL = 'ohmic dissipation'
#

MASK_Y1: Final[float] = 10**EIG_IM_LOG_MIN
MASK_Y2: Final[float] = - MASK_Y1


def wrapper_plot_eig(
        bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray]) \
        -> None:
    """A wrapper of a function to plot a figure of the dispersion
    relation (linear-linear)

    Parameters
    -----
    bundle : tuple of ndarray
        A tuple of results (linear-linear)

    """

    fig1: plt.Figure
    fig2: plt.Figure
    ax1: np.ndarray
    ax2: np.ndarray
    sc1: list
    sc2: list
    save_fig: set[int]
    (fig1, fig2, ax1, ax2, sc1, sc2), save_fig = plot_eig(bundle)

    for axis in (ax1[0], ax1[1], ax2[0], ax2[1]):
        axis.grid()
        axis.set_axisbelow(True)

        axis.set_xlim(ALPHA_INIT, ALPHA_END)
    #

    ax1[0].set_ylim(EIG_RE_INIT, EIG_RE_END)
    ax1[1].set_ylim(EIG_RE_INIT, EIG_RE_END)

    for axis in (ax1[0], ax1[1], ax2[0], ax2[1]):
        axis.set_xlabel(
            r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
            fontsize=16)
    #

    if save_fig & {1, 2}:
        ax1[0].set_ylabel(
            r'$\mathrm{Re}(\lambda)=\mathrm{Re}(\omega)/2\Omega_0$',
            fontsize=16)
        ax2[0].set_ylabel(
            r'$\mathrm{Im}(\lambda)=\mathrm{Im}(\omega)/2\Omega_0$',
            fontsize=16)
    else:
        ax1[0].set_ylabel(
            r'$\lambda=\omega/2\Omega_0$',
            fontsize=16)
    #

    ax1[0].set_title('Sinuous', fontsize=16)
    ax1[1].set_title('Varicose', fontsize=16)
    ax2[0].set_title('Sinuous', fontsize=16)
    ax2[1].set_title('Varicose', fontsize=16)

    for axis in (ax1[0], ax1[1], ax2[0], ax2[1]):
        axis.tick_params(labelsize=14)
        axis.minorticks_on()
    #

    if (not SWITCH_DISP_ETA) and (E_ETA == 0):
        fig1.suptitle(
            r'Dispersion relation '
            + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}',  fontsize=16)
        fig2.suptitle(
            r'Dispersion relation '
            + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}',  fontsize=16)
    else:
        fig1.suptitle(
            r'Dispersion relation '
            + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$E_\eta=$' + f' {E_ETA}',
            fontsize=16)
        fig2.suptitle(
            r'Dispersion relation '
            + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$E_\eta=$' + f' {E_ETA}',
            fontsize=16)
    #

    fig1.tight_layout()
    fig2.tight_layout()

    cbar_ax_1: plt.Axes
    cbar_ax_2: plt.Axes

    if SWITCH_COLOR in ('ene', 'ohm'):
        fig1.subplots_adjust(right=0.85)
        axpos = ax1[0].get_position()
        cbar_ax_1 = fig1.add_axes((0.88, axpos.y0, 0.01, axpos.height))
        if save_fig & {1, 2}:
            fig2.subplots_adjust(right=0.85)
            axpos = ax2[0].get_position()
            cbar_ax_2 = fig2.add_axes(
                (0.88, axpos.y0, 0.01, axpos.height))
        #
    #

    if SWITCH_COLOR == 'ene':
        cbar = fig1.colorbar(
            sc1[0], cax=cbar_ax_1, ticks=COLOR_TICKS_ATAN)
        cbar.ax.set_yticklabels(
            [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(label=CBAR_LABEL, size=16)

        if 1 in save_fig:
            cbar = fig2.colorbar(sc2[0], cax=cbar_ax_2,
                                 ticks=COLOR_TICKS_ATAN)
            cbar.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(label=CBAR_LABEL, size=16)
        elif 2 in save_fig:
            cbar = fig2.colorbar(sc2[1], cax=cbar_ax_2,
                                 ticks=COLOR_TICKS_ATAN)
            cbar.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(label=CBAR_LABEL, size=16)
        #
    elif SWITCH_COLOR == 'ohm':
        cbar = fig1.colorbar(sc1[0], cax=cbar_ax_1)
        cbar.set_label(label=CBAR_LABEL, size=16)

        if 1 in save_fig:
            cbar = fig2.colorbar(sc2[0], cax=cbar_ax_2)
            cbar.set_label(label=CBAR_LABEL, size=16)
        elif 2 in save_fig:
            cbar = fig2.colorbar(sc2[1], cax=cbar_ax_2)
            cbar.set_label(label=CBAR_LABEL, size=16)
        #
    #

    name_fig_full: str

    if (E_ETA != 0) and (CRITERION_Q > 0):
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX_1
    else:
        name_fig_full = NAME_FIG
    #

    if SWITCH_COLOR == 'blk':
        name_fig_full += NAME_FIG_SUFFIX_2[0]
    elif SWITCH_COLOR == 'ene':
        name_fig_full += NAME_FIG_SUFFIX_2[1]
    elif SWITCH_COLOR == 'ohm':
        name_fig_full += NAME_FIG_SUFFIX_2[2]
    #

    os.makedirs(PATH_DIR_FIG, exist_ok=True)

    name_fig_full_list: list[str] \
        = [name_fig_full + suffix for suffix in NAME_FIG_SUFFIX_3]
    path_fig: list[Path] \
        = [PATH_DIR_FIG / name for name in name_fig_full_list]

    fig1.savefig(path_fig[0], dpi=FIG_DPI)
    if save_fig & {1, 2}:
        fig2.savefig(path_fig[1], dpi=FIG_DPI)
    #
#


def plot_eig(bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                           np.ndarray, np.ndarray, np.ndarray]) \
        -> tuple[tuple[plt.Figure, plt.Figure,
                       np.ndarray, np.ndarray, list, list],
                 set[int]]:
    """Plots a figure of the dispersion relation (linear-linear)

    Parameters
    -----
    bundle : tuple of ndarray
        A tuple of results (linear-linear)

    Returns
    -----
    fig_bundle : tuple
        A tuple of figures
    save_fig : set of int
        The set of integers to determine whether to save a figure

    """

    lin_alpha: np.ndarray
    eig: np.ndarray
    mke: np.ndarray
    ohm: np.ndarray
    sym: np.ndarray
    lin_alpha, eig, mke, _, ohm, sym = bundle

    fig1: plt.Figure
    fig2: plt.Figure
    ax1: np.ndarray
    ax2: np.ndarray
    # real part
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 7))
    # imaginary part
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))

    sc1: list = [None, ] * 2
    sc2: list = [None, ] * 2

    save_fig: set[int] = set()

    cmap_min: float = math.nan
    cmap_max: float = math.nan
    if SWITCH_COLOR == 'ene':
        cmap_min = math.atan(STRETCH_ATAN * (0-0.5))
        cmap_max = math.atan(STRETCH_ATAN * (1-0.5))
    elif SWITCH_COLOR == 'ohm':
        cmap_min = 0
        cmap_max = OHM_MAX
    #

    alpha: float
    ones_alpha: np.ndarray

    dict_eig: dict[str, np.ndarray]

    scatter_color: np.ndarray

    for i_alpha in range(NUM_ALPHA):
        alpha = lin_alpha[i_alpha]

        ones_alpha = np.full(SIZE_MAT, alpha)

        dict_eig = pickup_eig(
            eig[i_alpha, :], mke[i_alpha, :], sym[i_alpha, :])

        if SWITCH_COLOR == 'blk':

            ax1[0].scatter(
                ones_alpha, dict_eig['s'].real, s=0.1, c='black')
            ax1[1].scatter(
                ones_alpha, dict_eig['v'].real, s=0.1, c='black')

            if E_ETA == 0:
                if False in np.isnan(dict_eig['s_u']):
                    ax1[0].scatter(ones_alpha, dict_eig['s_u'].real,
                                   s=0.2, c='red')
                    ax2[0].scatter(ones_alpha, dict_eig['s_u'].imag,
                                   s=0.2, c='red')
                    save_fig.add(1)
                #
                if False in np.isnan(dict_eig['v_u']):
                    ax1[1].scatter(ones_alpha, dict_eig['v_u'].real,
                                   s=0.2, c='red')
                    ax2[1].scatter(ones_alpha, dict_eig['v_u'].imag,
                                   s=0.2, c='red')
                    save_fig.add(2)
                #
            else:
                ax2[0].scatter(ones_alpha, dict_eig['s'].imag,
                               s=0.1, c='black')
                ax2[1].scatter(ones_alpha, dict_eig['v'].imag,
                               s=0.1, c='black')
                save_fig.union({1, 2})
            #

        elif SWITCH_COLOR in ('ene', 'ohm'):

            if SWITCH_COLOR == 'ene':
                scatter_color = np.arctan(
                    STRETCH_ATAN
                    * (mke[i_alpha, :]-0.5*np.ones(SIZE_MAT)))
            elif SWITCH_COLOR == 'ohm':
                scatter_color = ohm[i_alpha, :]
            #

            if E_ETA == 0:  # ene

                ax1[0].scatter(
                    ones_alpha, dict_eig['s_a'].real, s=0.05,
                    c=scatter_color,
                    cmap='jet', vmin=cmap_min, vmax=cmap_max)
                ax1[1].scatter(
                    ones_alpha, dict_eig['v_a'].real, s=0.05,
                    c=scatter_color,
                    cmap='jet', vmin=cmap_min, vmax=cmap_max)

                sc1[0] = ax1[0].scatter(
                    ones_alpha, dict_eig['s_na'].real, s=0.2,
                    c=scatter_color, cmap='jet', vmin=cmap_min,
                    vmax=cmap_max)
                ax1[1].scatter(
                    ones_alpha, dict_eig['v_na'].real, s=0.2,
                    c=scatter_color, cmap='jet', vmin=cmap_min,
                    vmax=cmap_max)

                if False in np.isnan(dict_eig['s_u']):
                    sc2[0] = ax2[0].scatter(
                        ones_alpha, dict_eig['s_u'].imag, s=0.1,
                        c=scatter_color, cmap='jet',
                        vmin=cmap_min, vmax=cmap_max)
                    save_fig.add(1)
                #
                if False in np.isnan(dict_eig['v_u']):
                    sc2[1] = ax2[1].scatter(
                        ones_alpha, dict_eig['v_u'].imag, s=0.1,
                        c=scatter_color, cmap='jet',
                        vmin=cmap_min, vmax=cmap_max)
                    save_fig.add(2)
                #
            else:
                sc1[0] = ax1[0].scatter(
                    ones_alpha, dict_eig['s'].real, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[1].scatter(
                    ones_alpha, dict_eig['v'].real, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)

                sc2[0] = ax2[0].scatter(
                    ones_alpha, dict_eig['s'].imag, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                sc2[1] = ax2[1].scatter(
                    ones_alpha, dict_eig['v'].imag, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                save_fig.union({1, 2})
            #
        #
    #

    fig_bundle: tuple[
        plt.Figure, plt.Figure, np.ndarray, np.ndarray, list, list] \
        = (fig1, fig2, ax1, ax2, sc1, sc2)

    return fig_bundle, save_fig
#


def wrapper_plot_eig_log(
        bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray]) \
        -> None:
    """A wrapper of a function to plot a figure of the dispersion
    relation (log-log)

    Parameters
    -----
    bundle_log : tuple of ndarray
        A tuple of results (log-log)

    """

    fig1: plt.Figure
    fig2: plt.Figure
    ax1: np.ndarray
    ax2: np.ndarray
    sc1: list
    sc2: list
    save_fig: set[int]
    (fig1, fig2, ax1, ax2, sc1, sc2), save_fig = plot_eig_log(bundle)

    ax_all: tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes,
                  plt.Axes, plt.Axes, plt.Axes, plt.Axes] \
        = (ax1[0, 0], ax1[0, 1], ax1[1, 0], ax1[1, 1],
           ax2[0, 0], ax2[0, 1], ax2[1, 0], ax2[1, 1])

    for axis in ax_all:
        axis.grid()
        axis.set_axisbelow(True)
    #

    for axis in (ax1[0, 0], ax1[0, 1], ax1[1, 0], ax1[1, 1]):
        axis.set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
        axis.set_ylim(10**EIG_RE_LOG_INIT, 10**EIG_RE_LOG_END)
    #
    for axis in (ax2[0, 0], ax2[0, 1], ax2[1, 0], ax2[1, 1]):
        axis.set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    #

    for axis in ax_all:
        axis.set_xscale('log')
    #
    for axis in (ax1[0, 0], ax1[0, 1], ax1[1, 0], ax1[1, 1]):
        axis.set_yscale('log')
    #
    for axis in (ax2[0, 0], ax2[0, 1], ax2[1, 0], ax2[1, 1]):
        axis.set_yscale('symlog', linthresh=10**EIG_IM_LOG_MIN)
    #

    ax1[1, 0].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    ax1[1, 1].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    ax2[1, 0].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    ax2[1, 1].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)

    if save_fig & {1, 2, 3, 4}:
        ax1[0, 0].set_ylabel(
            r'$|\mathrm{Re}(\lambda)|=|\mathrm{Re}(\omega)/2\Omega_0|$',
            fontsize=16)
        ax1[1, 0].set_ylabel(
            r'$|\mathrm{Re}(\lambda)|=|\mathrm{Re}(\omega)/2\Omega_0|$',
            fontsize=16)
        ax2[0, 0].set_ylabel(
            r'$\mathrm{Im}(\lambda)=\mathrm{Im}(\omega)/2\Omega_0$',
            fontsize=16)
        ax2[1, 0].set_ylabel(
            r'$\mathrm{Im}(\lambda)=\mathrm{Im}(\omega)/2\Omega_0$',
            fontsize=16)

        ax1[0, 0].set_title(
            r'Sinuous, Retrograde ($\mathrm{Re}(\lambda)<0$)',
            fontsize=16)
        ax1[0, 1].set_title(
            r'Sinuous, Prograde ($\mathrm{Re}(\lambda)>0$)',
            fontsize=16)
        ax1[1, 0].set_title(
            r'Varicose, Retrograde ($\mathrm{Re}(\lambda)<0$)',
            fontsize=16)
        ax1[1, 1].set_title(
            r'Varicose, Prograde ($\mathrm{Re}(\lambda)>0$)',
            fontsize=16)
        ax2[0, 0].set_title(
            r'Sinuous, Retrograde ($\mathrm{Re}(\lambda)<0$)',
            fontsize=16)
        ax2[0, 1].set_title(
            r'Sinuous, Prograde ($\mathrm{Re}(\lambda)>0$)',
            fontsize=16)
        ax2[1, 0].set_title(
            r'Varicose, Retrograde ($\mathrm{Re}(\lambda)<0$)',
            fontsize=16)
        ax2[1, 1].set_title(
            r'Varicose, Prograde ($\mathrm{Re}(\lambda)>0$)',
            fontsize=16)
    else:
        ax1[0, 0].set_ylabel(
            r'$|\lambda|=|\omega/2\Omega_0|$',
            fontsize=16)
        ax1[1, 0].set_ylabel(
            r'$|\lambda|=|\omega/2\Omega_0|$',
            fontsize=16)

        ax1[0, 0].set_title(
            r'Sinuous, Retrograde ($\lambda<0$)',
            fontsize=16)
        ax1[0, 1].set_title(
            r'Sinuous, Prograde ($\lambda>0$)',
            fontsize=16)
        ax1[1, 0].set_title(
            r'Varicose, Retrograde ($\lambda<0$)',
            fontsize=16)
        ax1[1, 1].set_title(
            r'Varicose, Prograde ($\lambda>0$)',
            fontsize=16)
    #

    for axis in ax_all:
        axis.tick_params(labelsize=12)
    #

    if (not SWITCH_DISP_ETA) and (E_ETA == 0):
        fig1.suptitle(
            r'Dispersion relation '
            + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}',  fontsize=16)
        fig2.suptitle(
            r'Dispersion relation '
            + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}',  fontsize=16)
    else:
        fig1.suptitle(
            r'Dispersion relation '
            + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$E_\eta=$' + f' {E_ETA}',
            fontsize=16)
        fig2.suptitle(
            r'Dispersion relation '
            + r'[$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$E_\eta=$' + f' {E_ETA}',
            fontsize=16)
    #

    fig1.tight_layout()
    fig2.tight_layout()

    cbar_ax_1: plt.Axes
    cbar_ax_2: plt.Axes

    if SWITCH_COLOR in ('ene', 'ohm'):
        fig1.subplots_adjust(right=0.85)
        expos1 = ax1[0, 0].get_position()
        cbar_ax_1 = fig1.add_axes(
            (0.88, expos1.y0, 0.01, expos1.height))
        if save_fig & {1, 2, 3, 4}:
            fig2.subplots_adjust(right=0.85)
            expos2 = ax2[0, 0].get_position()
            cbar_ax_2 = fig2.add_axes(
                (0.88, expos2.y0, 0.01, expos2.height))
        #
    #

    if SWITCH_COLOR == 'ene':

        cbar1 = fig1.colorbar(
            sc1[0], cax=cbar_ax_1, ticks=COLOR_TICKS_ATAN)
        cbar1.ax.set_yticklabels(
            [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
        cbar1.ax.tick_params(labelsize=14)
        cbar1.set_label(label=CBAR_LABEL, size=16)

        if 1 in save_fig:
            cbar2 = fig2.colorbar(sc2[0], cax=cbar_ax_2,
                                  ticks=COLOR_TICKS_ATAN)
            cbar2.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 2 in save_fig:
            cbar2 = fig2.colorbar(sc2[1], cax=cbar_ax_2,
                                  ticks=COLOR_TICKS_ATAN)
            cbar2.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 3 in save_fig:
            cbar2 = fig2.colorbar(sc2[2], cax=cbar_ax_2,
                                  ticks=COLOR_TICKS_ATAN)
            cbar2.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 4 in save_fig:
            cbar2 = fig2.colorbar(sc2[3], cax=cbar_ax_2,
                                  ticks=COLOR_TICKS_ATAN)
            cbar2.ax.set_yticklabels(
                [f'${i_ticks+0.5}$' for i_ticks in COLOR_TICKS])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        #
    elif SWITCH_COLOR == 'ohm':
        cbar1 = fig1.colorbar(sc1[0], cax=cbar_ax_1)
        cbar1.set_label(label=CBAR_LABEL, size=16)
        if 1 in save_fig:
            cbar2 = fig2.colorbar(sc2[0], cax=cbar_ax_2)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 2 in save_fig:
            cbar2 = fig2.colorbar(sc2[1], cax=cbar_ax_2)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 3 in save_fig:
            cbar2 = fig2.colorbar(sc2[2], cax=cbar_ax_2)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        elif 4 in save_fig:
            cbar2 = fig2.colorbar(sc2[3], cax=cbar_ax_2)
            cbar2.set_label(label=CBAR_LABEL, size=16)
        #
    #

    # ax1[0, 1].scatter(0.013, 0.00012, s=50, c='white', marker='*',
    #                   linewidth=0.5, edgecolors="black")
    # ax1[0, 1].scatter(0.013, 0.00025, s=50, c='white', marker='*',
    #                   linewidth=0.5, edgecolors="black")

    name_fig_full: str

    if (E_ETA != 0) and (CRITERION_Q > 0):
        name_fig_full = NAME_FIG + NAME_FIG_SUFFIX_1
    else:
        name_fig_full = NAME_FIG
    #

    if SWITCH_COLOR == 'blk':
        name_fig_full += NAME_FIG_SUFFIX_2[0]
    elif SWITCH_COLOR == 'ene':
        name_fig_full += NAME_FIG_SUFFIX_2[1]
    elif SWITCH_COLOR == 'ohm':
        name_fig_full += NAME_FIG_SUFFIX_2[2]
    #

    os.makedirs(PATH_DIR_FIG, exist_ok=True)

    name_fig_full_list: list[str] \
        = [name_fig_full + suffix for suffix in NAME_FIG_SUFFIX_4]
    path_fig: list[Path] \
        = [PATH_DIR_FIG / name for name in name_fig_full_list]

    fig1.savefig(path_fig[0], dpi=FIG_DPI)
    if save_fig & {1, 2, 3, 4}:
        fig2.savefig(path_fig[1], dpi=FIG_DPI)
    #
#


def plot_eig_log(bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray]) \
        -> tuple[tuple[plt.Figure, plt.Figure,
                       np.ndarray, np.ndarray, list, list],
                 set[int]]:
    """Plots a figure of the dispersion relation (log-log)

    Parameters
    -----
    bundle_log : tuple of ndarray
        A tuple of results (log-log)

    Returns
    -----
    fig_bundle : tuple
        A tuple of figures
    save_fig : set of int
        The set of integers to determine whether to save a figure

    """

    lin_alpha: np.ndarray
    eig: np.ndarray
    mke: np.ndarray
    ohm: np.ndarray
    sym: np.ndarray
    lin_alpha, eig, mke, _, ohm, sym = bundle

    fig1: plt.Figure
    fig2: plt.Figure
    ax1: np.ndarray
    ax2: np.ndarray
    # real part
    fig1, ax1 = plt.subplots(2, 2, figsize=(10, 10))
    # imaginary part
    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10))

    sc1: list = [None, ] * 4
    sc2: list = [None, ] * 4

    save_fig: set = set()

    cmap_min: float = math.nan
    cmap_max: float = math.nan
    if SWITCH_COLOR == 'ene':
        cmap_min = math.atan(STRETCH_ATAN * (0-0.5))
        cmap_max = math.atan(STRETCH_ATAN * (1-0.5))
    elif SWITCH_COLOR == 'ohm':
        cmap_min = 0
        cmap_max = OHM_LOG_MAX
    #

    alpha: float
    ones_alpha: np.ndarray

    dict_eig: dict[str, np.ndarray]

    scatter_color: np.ndarray

    for i_alpha in range(NUM_ALPHA_LOG):
        alpha = 10**lin_alpha[i_alpha]

        ones_alpha = np.full(SIZE_MAT, alpha)

        dict_eig = pickup_eig(
            eig[i_alpha, :], mke[i_alpha, :], sym[i_alpha, :])

        dict_eig['sr'] = -np.conjugate(dict_eig['sr'])
        dict_eig['vr'] = -np.conjugate(dict_eig['vr'])
        dict_eig['sr_u'] = -np.conjugate(dict_eig['sr_u'])
        dict_eig['vr_u'] = -np.conjugate(dict_eig['vr_u'])
        dict_eig['sr_a'] = -np.conjugate(dict_eig['sr_a'])
        dict_eig['vr_a'] = -np.conjugate(dict_eig['vr_a'])
        dict_eig['sr_na'] = -np.conjugate(dict_eig['sr_na'])
        dict_eig['vr_na'] = -np.conjugate(dict_eig['vr_na'])

        if SWITCH_COLOR == 'blk':

            ax1[0, 0].scatter(
                ones_alpha, dict_eig['sr'].real, s=0.1, c='black')
            ax1[0, 1].scatter(
                ones_alpha, dict_eig['sp'].real, s=0.1, c='black')
            ax1[1, 0].scatter(
                ones_alpha, dict_eig['vr'].real, s=0.1, c='black')
            ax1[1, 1].scatter(
                ones_alpha, dict_eig['vp'].real, s=0.1, c='black')

            if E_ETA == 0:
                if False in np.isnan(dict_eig['sr_u']):
                    ax1[0, 0].scatter(ones_alpha, dict_eig['sr_u'].real,
                                      s=0.2, c='red')
                    ax2[0, 0].scatter(ones_alpha, dict_eig['sr_u'].imag,
                                      s=0.2, c='red')
                    save_fig.add(1)
                #
                if False in np.isnan(dict_eig['sp_u']):
                    ax1[0, 1].scatter(ones_alpha, dict_eig['sp_u'].real,
                                      s=0.2, c='red')
                    ax2[0, 1].scatter(ones_alpha, dict_eig['sp_u'].imag,
                                      s=0.2, c='red')
                    save_fig.add(2)
                #
                if False in np.isnan(dict_eig['vr_u']):
                    ax1[1, 0].scatter(ones_alpha, dict_eig['vr_u'].real,
                                      s=0.2, c='red')
                    ax2[1, 0].scatter(ones_alpha, dict_eig['vr_u'].imag,
                                      s=0.2, c='red')
                    save_fig.add(3)
                #
                if False in np.isnan(dict_eig['vp_u']):
                    ax1[1, 1].scatter(ones_alpha, dict_eig['vp_u'].real,
                                      s=0.2, c='red')
                    ax2[1, 1].scatter(ones_alpha, dict_eig['vp_u'].imag,
                                      s=0.2, c='red')
                    save_fig.add(4)
                #
            else:
                ax2[0, 0].scatter(
                    ones_alpha, dict_eig['sr'].imag, s=0.1, c='black')
                ax2[0, 1].scatter(
                    ones_alpha, dict_eig['sp'].imag, s=0.1, c='black')
                ax2[1, 0].scatter(
                    ones_alpha, dict_eig['vr'].imag, s=0.1, c='black')
                ax2[1, 1].scatter(
                    ones_alpha, dict_eig['vp'].imag, s=0.1, c='black')
                save_fig.union({1, 2, 3, 4})
            #

        elif SWITCH_COLOR in ('ene', 'ohm'):

            if SWITCH_COLOR == 'ene':
                scatter_color = np.arctan(
                    STRETCH_ATAN
                    * (mke[i_alpha, :]-0.5*np.ones(SIZE_MAT)))
            elif SWITCH_COLOR == 'ohm':
                scatter_color = ohm[i_alpha, :]
            #

            if E_ETA == 0:  # ene
                ax1[0, 0].scatter(
                    ones_alpha, dict_eig['sr_a'].real, s=0.05,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[0, 1].scatter(
                    ones_alpha, dict_eig['sp_a'].real, s=0.05,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[1, 0].scatter(
                    ones_alpha, dict_eig['vr_a'].real, s=0.05,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[1, 1].scatter(
                    ones_alpha, dict_eig['vp_a'].real, s=0.05,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)

                sc1[0] = ax1[0, 0].scatter(
                    ones_alpha, dict_eig['sr_na'].real, s=0.2,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[0, 1].scatter(
                    ones_alpha, dict_eig['sp_na'].real, s=0.2,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[1, 0].scatter(
                    ones_alpha, dict_eig['vr_na'].real, s=0.2,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[1, 1].scatter(
                    ones_alpha, dict_eig['vp_na'].real, s=0.2,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)

                if False in np.isnan(dict_eig['sr_u']):
                    sc2[0] = ax2[0, 0].scatter(
                        ones_alpha, dict_eig['sr_u'].imag, s=0.1,
                        c=scatter_color, cmap='jet',
                        vmin=cmap_min, vmax=cmap_max)
                    save_fig.add(1)
                #
                if False in np.isnan(dict_eig['sp_u']):
                    sc2[1] = ax2[0, 1].scatter(
                        ones_alpha, dict_eig['sp_u'].imag, s=0.1,
                        c=scatter_color, cmap='jet',
                        vmin=cmap_min, vmax=cmap_max)
                    save_fig.add(2)
                #
                if False in np.isnan(dict_eig['vr_u']):
                    sc2[2] = ax2[1, 0].scatter(
                        ones_alpha, dict_eig['vr_u'].imag, s=0.1,
                        c=scatter_color, cmap='jet',
                        vmin=cmap_min, vmax=cmap_max)
                    save_fig.add(3)
                #
                if False in np.isnan(dict_eig['vp_u']):
                    sc2[3] = ax2[1, 1].scatter(
                        ones_alpha, dict_eig['vp_u'].imag, s=0.1,
                        c=scatter_color, cmap='jet',
                        vmin=cmap_min, vmax=cmap_max)
                    save_fig.add(4)
                #
            else:
                sc1[0] = ax1[0, 0].scatter(
                    ones_alpha, dict_eig['sr'].real, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[0, 1].scatter(
                    ones_alpha, dict_eig['sp'].real, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[1, 0].scatter(
                    ones_alpha, dict_eig['vr'].real, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                ax1[1, 1].scatter(
                    ones_alpha, dict_eig['vp'].real, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)

                sc2[0] = ax2[0, 0].scatter(
                    ones_alpha, dict_eig['sr'].imag, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                sc2[1] = ax2[0, 1].scatter(
                    ones_alpha, dict_eig['sp'].imag, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                sc2[2] = ax2[1, 0].scatter(
                    ones_alpha, dict_eig['vr'].imag, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                sc2[3] = ax2[1, 1].scatter(
                    ones_alpha, dict_eig['vp'].imag, s=0.1,
                    c=scatter_color, cmap='jet',
                    vmin=cmap_min, vmax=cmap_max)
                save_fig.add({1, 2, 3, 4})
            #
        #
    #

    mask_x: np.ndarray = 10**lin_alpha
    ax2[0, 0].fill_between(mask_x, MASK_Y1, MASK_Y2, facecolor='grey')
    ax2[0, 1].fill_between(mask_x, MASK_Y1, MASK_Y2, facecolor='grey')
    ax2[1, 0].fill_between(mask_x, MASK_Y1, MASK_Y2, facecolor='grey')
    ax2[1, 1].fill_between(mask_x, MASK_Y1, MASK_Y2, facecolor='grey')

    fig_bundle: tuple[
        plt.Figure, plt.Figure, np.ndarray, np.ndarray, list, list] \
        = (fig1, fig2, ax1, ax2, sc1, sc2)

    return fig_bundle, save_fig
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    logging.basicConfig(level=logging.INFO)
    logger: logging.Logger = logging.getLogger(__name__)

    if True not in SWITCH_PLOT:
        logger.info('No plotted figures')
        sys.exit()
    #
    if SWITCH_COLOR not in ('blk', 'ene', 'ohm'):
        logger.warning('Invalid value for \'SWITCH_COLOR\'')
        sys.exit()
    #
    if (E_ETA == 0) and (SWITCH_COLOR == 'ohm'):
        logger.warning('Meaningless figures are plotted')
        sys.exit()
    #

    results: tuple[np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray, np.ndarray]
    results_log: tuple[np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray, np.ndarray, np.ndarray]
    results, results_log \
        = wrapper_load_results(SWITCH_PLOT, M_ORDER, E_ETA, N_T)

    plt.rcParams['text.usetex'] = True

    if SWITCH_PLOT[0]:

        if (E_ETA != 0) and (CRITERION_Q > 0):
            results = screening_eig_q(CRITERION_Q, results)
        #

        ALPHA_INIT: float
        ALPHA_END: float
        NUM_ALPHA: int
        OHM_MAX: float
        ALPHA_INIT, ALPHA_END, NUM_ALPHA, OHM_MAX \
            = pickup_param(results)

        wrapper_plot_eig(results)
    #
    if SWITCH_PLOT[1]:

        if (E_ETA != 0) and (CRITERION_Q > 0):
            results_log = screening_eig_q(CRITERION_Q, results_log)
        #

        ALPHA_LOG_INIT: float
        ALPHA_LOG_END: float
        NUM_ALPHA_LOG: int
        OHM_LOG_MAX: float
        ALPHA_LOG_INIT, ALPHA_LOG_END, NUM_ALPHA_LOG, OHM_LOG_MAX \
            = pickup_param(results_log)

        wrapper_plot_eig_log(results_log)
    #

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
