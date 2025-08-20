"""A Python script to calculate the dispersion relation of
two-dimensional (2D) ideal magnetohydrodynamic (MHD) waves on a rotating
sphere under the Malkus background field, B_phi = B_0 sin(theta).

This script can create up to three figures: a linear-linear plot of
the dispersion relation, a plot showing energy partitioning for various
eigenmodes, and a log-log plot of the dispersion relation.

Parameters
----------
M_ORDER : int
    The zonal wavenumber (order).

Warnings
--------
No plotted figures
    If all of the boolean values to switch whether to plot figures or
    not are False.

Notes
-----
All other parameters aside from command line arguments are described
within the script.

References
----------
[1] Ryosuke Nakashima, Shigeo Yoshida, Two-dimensional ideal
magnetohydrodynamic waves on a rotating sphere under a non-Malkus field:
I. Continuous spectrum and its ray-theoretical interpretation.
Geophysical & Astrophysical Fluid Dynamics 118(5-6), 387-440 (2024).
doi: 10.1080/03091929.2024.2384388

Examples
--------
Run the script with the default value of M_ORDER:
    $ python3 mhd2dsphere_malkus.py
Run the script with a specified value (say M_ORDER = 2):
    $ python3 mhd2dsphere_malkus.py 2
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from package_common.common_types import (ArrayFloat, ArrayInt, Artist, Final,
                                         Legend)
from package_common.default_logger import DefaultLogger
from package_common.default_plotter import (DefaultGridPlotter, DefaultPlotter,
                                            create_plotter)
from package_common.default_timer import DefaultTimer
from package_common.input_helper import input_value
from package_common.name_utils import create_function_name_progress_bar
from package_common.progress_bar import ProgressBar

# ========== Parameters ========== #

# The boolean values to switch whether to plot figures or not
# SWITCH_PLOT[0]: The linear-linear plot of the dispersion relation
# SWITCH_PLOT[1]: The plot showing energy partitioning
# SWITCH_PLOT[2]: The log-log plot of the dispersion relation
SWITCH_PLOT: Final[tuple[bool, bool, bool]] = (True, True, True)

# The zonal wavenumber (order)
M_ORDER: Final[int] = input_value(1, int)

# Degrees
N_INIT: Final[int] = M_ORDER  # M_ORDER <= N_INIT
N_STEP: Final[int] = 1
N_END: Final[int] = 10

# The range of the Lehnert number
# linear
ALPHA_INIT: Final[float] = 0
ALPHA_STEP: Final[float] = 0.001
ALPHA_END: Final[float] = 1
# log
ALPHA_LOG_INIT: Final[float] = -4
ALPHA_LOG_STEP: Final[float] = 0.01
ALPHA_LOG_END: Final[float] = 2

# The range of eigenvalues
# linear
EIG_INIT: Final[float] = -2
EIG_END: Final[float] = 2
# log
EIG_LOG_INIT: Final[float] = -6
EIG_LOG_END: Final[float] = 2

# The range of energy partitioning
ENERGY_INIT: Final[float] = 0
ENERGY_END: Final[float] = 1

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] = Path('.') / 'fig' / 'MHD2Dsphere_malkus'
NAME_FIG_1: Final[str] = f'MHD2Dsphere_malkus_m{M_ORDER}_eig.png'
NAME_FIG_2: Final[str] = f'MHD2Dsphere_malkus_m{M_ORDER}_ene.png'
NAME_FIG_3: Final[str] = f'MHD2Dsphere_malkus_m{M_ORDER}_eiglog.png'
FIG_DPI: Final[int] = 600

# ================================ #

NAMES_MODE: Final[tuple[str, str]] = ('fMR', 'sMR')
NUM_MODE: Final[int] = len(NAMES_MODE)

NUM_N: Final[int] = 1 + int((N_END-N_INIT)/N_STEP)
NUM_ALPHA: Final[int] = 1 + int((ALPHA_END-ALPHA_INIT)/ALPHA_STEP)
NUM_ALPHA_LOG: Final[int] \
    = 1 + int((ALPHA_LOG_END-ALPHA_LOG_INIT)/ALPHA_LOG_STEP)

LIN_N: Final[ArrayInt] = np.linspace(
    N_INIT, N_END, NUM_N, dtype=np.int_)
LIN_ALPHA: Final[ArrayFloat] = np.linspace(
    ALPHA_INIT, ALPHA_END, NUM_ALPHA, dtype=np.float64)
LIN_ALPHA_LOG: Final[ArrayFloat] = np.linspace(
    ALPHA_LOG_INIT, ALPHA_LOG_END, NUM_ALPHA_LOG, dtype=np.float64)


def wrapper_eigene() -> tuple[ArrayFloat,
                              ArrayFloat,
                              ArrayFloat]:
    """Calculate the dispersion relation and energy partitioning.

    Returns
    -------
    eig : ArrayFloat
        Eigenvalues (linear-linear).
    ene : ArrayFloat
        Energy partitioning.
    eig_log : ArrayFloat
        Eigenvalues (log-log).
    """

    eig: ArrayFloat \
        = np.empty((NUM_N, NUM_ALPHA, NUM_MODE), dtype=np.float64)
    ene: ArrayFloat \
        = np.empty((NUM_N, NUM_ALPHA_LOG, NUM_MODE), dtype=np.float64)
    eig_log: ArrayFloat \
        = np.empty((NUM_N, NUM_ALPHA_LOG, NUM_MODE), dtype=np.float64)

    n_degree: int
    alpha: float

    progress_bar: ProgressBar = create_function_name_progress_bar(NUM_N)
    progress_bar.start()
    for i_n in range(NUM_N):
        n_degree = LIN_N[i_n]

        if SWITCH_PLOT[0]:
            for i_alpha in range(NUM_ALPHA):
                alpha = LIN_ALPHA[i_alpha]

                for name_mode in NAMES_MODE:
                    eig[i_n, i_alpha, NAMES_MODE.index(name_mode)] \
                        = calc_eig(n_degree, alpha, name_mode)

        if (SWITCH_PLOT[1] or SWITCH_PLOT[2]):
            for i_alpha in range(NUM_ALPHA_LOG):
                alpha = 10**LIN_ALPHA_LOG[i_alpha]

                for name_mode in NAMES_MODE:
                    ene[i_n, i_alpha, NAMES_MODE.index(name_mode)] \
                        = calc_ene(n_degree, alpha, name_mode)
                    eig_log[i_n, i_alpha, NAMES_MODE.index(name_mode)] \
                        = calc_eig(n_degree, alpha, name_mode)

        progress_bar.update(i_n)

    return eig, ene, eig_log


def calc_eig(n_degree: int,
             alpha: float,
             name_mode: str) -> float:
    """Calculate the dispersion relation.

    Parameters
    ----------
    n_degree : int
        The degree of the associated Legendre polynomial.
    alpha : float
        The Lehnert number.
    name_mode : str
        'fMR' or 'sMR'.

    Returns
    -------
    float
        An eigenvalue.

    Warnings
    --------
    Invalid ID
        If name_mode is neither 'fMR' nor 'sMR'.

    Notes
    -----
    If n_degree = 0, eig is set to 0. This function is based on eq. (1)
    in Nakashima & Yoshida (2024)[1]_.
    """

    if n_degree == 0:
        return 0

    nn1: int = n_degree * (n_degree+1)
    sqrt_part: float = math.sqrt(1 + 4*(alpha**2)*nn1*(nn1-2))

    if name_mode == 'fMR':
        return (-M_ORDER-M_ORDER*sqrt_part) / (2*nn1)
    if name_mode == 'sMR':
        return (-M_ORDER+M_ORDER*sqrt_part) / (2*nn1)

    DefaultLogger(__name__).error('Invalid ID')
    sys.exit(1)


def calc_ene(n_degree: int,
             alpha: float,
             name_mode: str) -> float:
    """Calculate energy partitioning.

    Parameters
    ----------
    n_degree : int
        The degree of the associated Legendre polynomial.
    alpha : float
        The Lehnert number.
    name_mode : str
        'fMR' or 'sMR'.

    Returns
    -------
    float
        Energy partitioning.

    Notes
    -----
    If n_degree = 0, ene is set to 0. If alpha = 0, ene is set to 0 for
    fast MR waves and 1 for slow MR waves. This function is based on the
    equation in the caption of Fig. 3 in Nakashima & Yoshida (2024)[1]_.
    """

    if n_degree == 0:
        return 0

    if alpha == 0:
        if name_mode == 'fMR':
            return 0
        if name_mode == 'sMR':
            return 1

    ma2: float = (M_ORDER**2) * (alpha**2)
    lambda_mr: float = calc_eig(n_degree, alpha, name_mode)
    return (lambda_mr**2) / ((lambda_mr**2)+ma2)


def plot_eig(eig: ArrayFloat) -> None:
    """Create the linear-linear plot of the dispersion relation.

    Parameters
    ----------
    eig : ArrayFloat
        Eigenvalues.
    """

    plotter: DefaultPlotter = create_plotter(1, 1, figsize=(5, 7))

    i_n: int
    for i_n_inv in range(NUM_N):
        i_n = NUM_N - 1 - i_n_inv

        if i_n not in (0, NUM_N-1):
            plotter.axes.plot(
                LIN_ALPHA, eig[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0])
            plotter.axes.plot(
                LIN_ALPHA, eig[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1])
        else:
            plotter.axes.plot(
                LIN_ALPHA, eig[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0],
                label=r'$n=$'+f' {N_INIT+i_n} fast MR')
            plotter.axes.plot(
                LIN_ALPHA, eig[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1],
                label=r'$n=$'+f' {N_INIT+i_n} slow MR')

    plotter.axes.set_xlim(ALPHA_INIT, ALPHA_END)
    plotter.axes.set_ylim(EIG_INIT, EIG_END)

    plotter.axes.set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    plotter.axes.set_ylabel(r'$\lambda=\omega/2\Omega_0$', fontsize=16)
    plotter.axes.set_title(
        r'Dispersion relation [$B_{0\phi}=B_0\sin\theta$] : $m=$'
        + f' {M_ORDER}\n', fontsize=16)

    handles: list[Artist]
    labels: list[str]
    [handles, labels] = plotter.axes.get_legend_handles_labels()
    order_leg: list[int] = [3, 1, 2, 0]
    handles = [handles[i_handle] for i_handle in order_leg]
    labels = [labels[i_label] for i_label in order_leg]
    leg: Legend
    if M_ORDER >= 3:
        leg = plotter.axes.legend(
            handles=handles, labels=labels, loc='center right',
            fontsize=14)
    else:
        leg = plotter.axes.legend(
            handles=handles, labels=labels, loc='lower left',
            fontsize=14)

    leg.get_frame().set_alpha(1)

    plotter.axes.tick_params(labelsize=14)

    plotter.save(PATH_DIR_FIG, NAME_FIG_1, FIG_DPI)


def plot_ene(ene: ArrayFloat) -> None:
    """Create the plot showing the energy partitioning for various
    eigenmodes.

    Parameters
    ----------
    ene : ArrayFloat
        Energy partitioning.
    """

    plotter: DefaultPlotter = create_plotter(1, 1, figsize=(5, 5))

    i_n: int
    for i_n_inv in range(NUM_N):
        i_n = NUM_N - 1 - i_n_inv

        if (M_ORDER == 1) and (i_n == 0):
            plotter.axes.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1], linewidth=3)

        if i_n not in (0, NUM_N-1):
            plotter.axes.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0])
            plotter.axes.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1])
        else:
            plotter.axes.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0],
                label=r'$n=$'+f' {N_INIT+i_n} fast MR')
            plotter.axes.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1],
                label=r'$n=$'+f' {N_INIT+i_n} slow MR')

    plotter.axes.set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    plotter.axes.set_ylim(ENERGY_INIT, ENERGY_END)

    plotter.axes.set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    plotter.axes.set_ylabel(
        r'$\mathrm{MKE}/(\mathrm{MKE}+\mathrm{MME})$', fontsize=16)
    plotter.axes.set_title(
        r'Energy partitioning [$B_{0\phi}=B_0\sin\theta$] : $m=$'
        + f' {M_ORDER}\n', fontsize=16)

    handles: list[Artist]
    labels: list[str]
    [handles, labels] = plotter.axes.get_legend_handles_labels()
    num_labels = len(labels)
    # Default order: [2, 0, 3, 1], but adjust if fewer labels
    if num_labels >= 4:
        order_leg: list[int] = [2, 0, 3, 1]
    else:
        order_leg: list[int] = list(range(num_labels))
    handles = [handles[i_handle] for i_handle in order_leg]
    labels = [labels[i_label] for i_label in order_leg]
    leg: Legend = plotter.axes.legend(
        handles=handles, labels=labels,
        loc='upper right', fontsize=12, bbox_to_anchor=(1.1, 1))
    leg.get_frame().set_alpha(1)

    plotter.axes.tick_params(labelsize=13)

    plotter.save(PATH_DIR_FIG, NAME_FIG_2, FIG_DPI)


def plot_eig_log(eig_log: ArrayFloat) -> None:
    """Create the log-log plot of the dispersion relation.

    Parameters
    ----------
    eig_log : ArrayFloat
        Eigenvalues.
    """

    plotter: DefaultGridPlotter = create_plotter(1, 2, figsize=(10, 5))

    i_n: int
    for i_n_inv in range(NUM_N):
        i_n = NUM_N - 1 - i_n_inv

        if (M_ORDER == 1) and (i_n == 0):
            plotter.axes[0].loglog(
                10**LIN_ALPHA_LOG,
                -eig_log[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0], label=r'$n=$ 1 fast MR')
            plotter.axes[1].loglog(
                10**LIN_ALPHA_LOG,
                eig_log[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1], label=r'$n=$ 2 slow MR')
        elif i_n not in (0, NUM_N-1):
            plotter.axes[0].loglog(
                10**LIN_ALPHA_LOG,
                -eig_log[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0])
            plotter.axes[1].loglog(
                10**LIN_ALPHA_LOG,
                eig_log[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1])
        else:
            plotter.axes[0].loglog(
                10**LIN_ALPHA_LOG,
                -eig_log[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0],
                label=r'$n=$'+f' {N_INIT+i_n} fast MR')
            plotter.axes[1].loglog(
                10**LIN_ALPHA_LOG,
                eig_log[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1],
                label=r'$n=$'+f' {N_INIT+i_n} slow MR')

    plotter.axes[0].set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    plotter.axes[1].set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    plotter.axes[0].set_ylim(10**EIG_LOG_INIT, 10**EIG_LOG_END)
    plotter.axes[1].set_ylim(10**EIG_LOG_INIT, 10**EIG_LOG_END)

    plotter.axes[0].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    plotter.axes[1].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    plotter.axes[0].set_ylabel(r'$|\lambda|=|\omega/2\Omega_0|$', fontsize=16)
    plotter.axes[0].set_title(r'Retrograde ($\lambda<0$)', fontsize=16)
    plotter.axes[1].set_title(r'Prograde ($\lambda>0$)', fontsize=16)

    handles: list[list] = [[], []]
    labels: list[list] = [[], []]

    [handles[0], labels[0]] \
        = plotter.axes[0].get_legend_handles_labels()
    leg1: Legend = plotter.axes[0].legend(
        handles=handles[0][::-1], labels=labels[0][::-1],
        loc='lower right', fontsize=14)
    leg1.get_frame().set_alpha(1)

    [handles[1], labels[1]] \
        = plotter.axes[1].get_legend_handles_labels()
    leg2: Legend = plotter.axes[1].legend(
        handles=handles[1][::-1], labels=labels[1][::-1],
        loc='lower right', fontsize=14)
    leg2.get_frame().set_alpha(1)

    plotter.axes[0].tick_params(labelsize=13)
    plotter.axes[1].tick_params(labelsize=13)

    plotter.fig.suptitle(
        r'Dispersion relation [$B_{0\phi}=B_0\sin\theta$] : $m=$'
        + f' {M_ORDER}', fontsize=16)

    plotter.save(PATH_DIR_FIG, NAME_FIG_3, FIG_DPI)


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    if not any(SWITCH_PLOT):
        DefaultLogger(__name__).warning('No plotted figures')
        sys.exit(0)

    data: tuple[ArrayFloat,
                ArrayFloat,
                ArrayFloat] = wrapper_eigene()

    if SWITCH_PLOT[0]:
        plot_eig(data[0])

    if SWITCH_PLOT[1]:
        plot_ene(data[1])

    if SWITCH_PLOT[2]:
        plot_eig_log(data[2])

    timer.end()

    plt.show()
