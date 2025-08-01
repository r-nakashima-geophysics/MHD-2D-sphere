"""A code for 2D MHD waves on a rotating sphere under the non-Malkus
field B_phi = B_0 sin(theta) cos(theta)

Plots a figure displaying the dependence of eigenvalues on the
truncation degree.

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

import os
from pathlib import Path
from time import perf_counter
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from package_mhd2dsphere.solve_eig import wrapper_solve_eig

# ========== Parameters ==========

# The boolean value to switch whether to display the value of the
# magnetic Ekman number when E_ETA = 0
SWITCH_DISP_ETA: Final[bool] = False

# The zonal wavenumber (order)
M_ORDER: Final[int] = 1

# The Lehnert number
ALPHA: Final[float] = 1

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The truncation degree
N_T_1: Final[int] = 2000
N_T_2: Final[int] = 1999

# Criteria for convergence
# degree
N_C_1: Final[int] = int(N_T_1/2)
N_C_2: Final[int] = int(N_T_2/2)
# ratio
R_C: Final[float] = 100

# The range of the eigenvalue number
NUMEIG_INIT: Final[int] = 1960
NUMEIG_END: Final[int] = 2040

# The range of eigenvalues
# log, real part
EIG_RE_LOG_MAX: Final[float] = -1.5
EIG_RE_LOG_MIN: Final[float] = -4

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_sincos_degree'
NAME_FIG: Final[str] = 'MHD2Dsphere_sincos_degree' \
    + f'_m{M_ORDER}a{ALPHA}E{E_ETA}N{N_T_1}vsN{N_T_2}.png'
FIG_DPI: Final[int] = 600

# ================================

CRITERION_C_1: Final[tuple[int, float]] = (N_C_1, R_C)
CRITERION_C_2: Final[tuple[int, float]] = (N_C_2, R_C)

SIZE_SUBMAT_1: Final[int] = N_T_1 - M_ORDER + 1
SIZE_MAT_1: Final[int] = 2 * SIZE_SUBMAT_1

SIZE_SUBMAT_2: Final[int] = N_T_2 - M_ORDER + 1
SIZE_MAT_2: Final[int] = 2 * SIZE_SUBMAT_2

MASK_X: Final[list[int]] \
    = list(range(1, max(SIZE_MAT_1, SIZE_MAT_2) + 1))
MASK_Y1: Final[float] = 10**EIG_RE_LOG_MIN
MASK_Y2: Final[float] = - MASK_Y1


def plot_dependdegree(
        bundle_with_vec_1: tuple[np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray],
        bundle_with_vec_2: tuple[np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray]) -> None:
    """Plots a figure displaying the dependence of eigenvalues on the
    truncation degree

    Parameters
    ----------
    bundle_with_vec_1 : tuple of ndarray
        A tuple of results
    bundle_with_vec_2 : tuple of ndarray
        A tuple of results

    """

    eig_1: np.ndarray = bundle_with_vec_1[2]
    eig_2: np.ndarray = bundle_with_vec_2[2]

    fig: plt.Figure
    axis: plt.Axes
    fig, axis = plt.subplots(figsize=(5, 5))

    i_mode_1: list[int] = list(range(1, SIZE_MAT_1 + 1))
    i_mode_2: list[int] = list(range(1, SIZE_MAT_2 + 1))

    axis.scatter(i_mode_1, eig_1.real, s=5, marker='s',
                 facecolors='none', edgecolors='red',
                 label=r'$N_\mathrm{t}=$' + f' {N_T_1}')
    axis.scatter(i_mode_2, eig_2.real, s=5, color='blue',
                 label=r'$N_\mathrm{t}=$' + f' {N_T_2}')

    axis.fill_between(MASK_X, MASK_Y1, MASK_Y2, facecolor='grey')

    axis.grid()
    axis.set_axisbelow(True)

    axis.set_xlim(NUMEIG_INIT, NUMEIG_END)
    axis.set_ylim(-10**EIG_RE_LOG_MAX, 10**EIG_RE_LOG_MAX)

    axis.set_yscale('symlog', linthresh=10**EIG_RE_LOG_MIN)

    axis.set_xlabel('eigenvalue number', fontsize=16)
    axis.set_ylabel(
        r'$\mathrm{Re}(\lambda)=\mathrm{Re}(\omega)/2\Omega_0$',
        fontsize=16)

    if (not SWITCH_DISP_ETA) and (E_ETA == 0):
        axis.set_title(
            r'Eigenvalues [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$'
            + f' {ALPHA}\n',
            fontsize=14)
    else:
        axis.set_title(
            r'Eigenvalues [$B_{0\phi}=B_0\sin\theta\cos\theta$] : '
            + r'$m=$' + f' {M_ORDER}, ' + r'$|\alpha|=$' + f' {ALPHA}, '
            + r'$E_\eta=$' + f' {E_ETA}\n',
            fontsize=12)

    leg: plt.Legend = axis.legend(loc='best', fontsize=14)
    leg.get_frame().set_alpha(1)

    axis.tick_params(labelsize=14)

    fig.tight_layout()

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / NAME_FIG
    fig.savefig(path_fig, dpi=FIG_DPI)
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    results_with_vec_1: tuple[np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray] \
        = wrapper_solve_eig(
        M_ORDER, ALPHA, E_ETA, SIZE_SUBMAT_1, CRITERION_C_1)
    results_with_vec_2: tuple[np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray] \
        = wrapper_solve_eig(
        M_ORDER, ALPHA, E_ETA, SIZE_SUBMAT_2, CRITERION_C_2)

    plt.rcParams['text.usetex'] = True

    plot_dependdegree(results_with_vec_1, results_with_vec_2)

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
