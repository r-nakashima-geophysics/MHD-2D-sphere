"""A Python script to calculate the dispersion relation of
two-dimensional (2D) magnetohydrodynamic (MHD) waves on a rotating
sphere under the non-Malkus toroidal background field, B_phi = B_0
sin(theta) cos(theta).

This script outputs npz files of results (alpha, eigenvalue, mean
kinetic energy, mean magnetic energy, ohmic dissipation, and the
symmetry of eigenmodes).

Parameters
----------
M_ORDER : int
    Zonal wavenumber (order).

Warnings
----------
No saved file
    If all of the boolean values to switch whether to calculate are
    False.

Notes
----------
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
----------
Run the script with the default value of M_ORDER:
    $ python3 mhd2dsphere_sincos.py
Run the script with a specified value (say M_ORDER = 2):
    $ python3 mhd2dsphere_sincos.py 2
"""

import inspect
import os
import sys
from pathlib import Path

import numpy as np

from package_common.common_types import (ArrayComplex, ArrayFloat, ArrayStr,
                                         Final)
from package_common.default_logger import DefaultLogger
from package_common.default_timer import DefaultTimer
from package_common.input_helper import input_value
from package_mhd2dsphere.make_mat import make_mat, make_submat_sincos
from package_mhd2dsphere.solve_eig import solve_eig

# ========== Parameters ========== #

# The boolean values to switch whether to calculate
# SWITCH_CALC[0]: The dispersion relation for the linear-linear plot
# SWITCH_CALC[1]: The dispersion relation for the log-log plot
SWITCH_CALC: Final[tuple[bool, bool]] = (True, True)

# Zonal wavenumber (order)
M_ORDER: Final[int] = input_value(1, int)

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The truncation degree
N_T: Final[int] = 2000

# The criterion for convergence
# degree
N_C: Final[int] = int(N_T/2)
# ratio
R_C: Final[float] = 100

# The range of the Lehnert number
# linear
ALPHA_INIT: Final[float] = 0
ALPHA_STEP: Final[float] = 0.001
ALPHA_END: Final[float] = 1
# log
ALPHA_LOG_INIT: Final[float] = -4
ALPHA_LOG_STEP: Final[float] = 0.01
ALPHA_LOG_END: Final[float] = 2

# The paths and filenames of outputs
PATH_DIR: Final[Path] \
    = Path('.') / 'output' / 'MHD2Dsphere_sincos'
NAME_FILE: Final[str] \
    = f'MHD2Dsphere_sincos_m{M_ORDER}E{E_ETA}N{N_T}'
NAME_FILE_SUFFIX: Final[tuple[str, str]] = ('.npz', '_log.npz')

# ================================

CRITERION_C: dict[str, int | float] = {
    'degree': N_C,
    'ratio': R_C
}

NUM_ALPHA: Final[int] \
    = 1 + int((ALPHA_END-ALPHA_INIT)/ALPHA_STEP)
NUM_ALPHA_LOG: Final[int] \
    = 1 + int((ALPHA_LOG_END-ALPHA_LOG_INIT)/ALPHA_LOG_STEP)

LIN_ALPHA: Final[ArrayFloat] \
    = np.linspace(ALPHA_INIT, ALPHA_END, NUM_ALPHA)
LIN_ALPHA_LOG: Final[ArrayFloat] \
    = np.linspace(ALPHA_LOG_INIT, ALPHA_LOG_END, NUM_ALPHA_LOG)

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT


def wrapper_solve_eig_for_alpha() -> tuple[tuple[ArrayComplex,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayStr],
                                           tuple[ArrayComplex,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayStr]]:
    """Solve the eigenvalue problem for a given alpha.

    Returns
    ----------
    results : tuple[ArrayComplex, ArrayFloat, ArrayFloat, ArrayFloat,
    ArrayStr]
        The tuple of results (linear-linear).
    results_log : tuple[ArrayComplex, ArrayFloat, ArrayFloat,
    ArrayFloat, ArrayStr]
        The tuple of results (log-log).
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    submatrices: tuple[ArrayFloat,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayFloat] \
        = make_submat_sincos(M_ORDER, SIZE_SUBMAT)

    results: tuple[ArrayComplex,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayStr]
    results_log: tuple[ArrayComplex,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayStr]

    eig: ArrayComplex
    mke: ArrayFloat
    mme: ArrayFloat
    ohm: ArrayFloat
    sym: ArrayStr

    alpha: float
    mat: ArrayComplex
    eig_valvec: ArrayComplex
    phys_qtys: tuple[ArrayFloat,
                     ArrayFloat,
                     ArrayFloat,
                     ArrayStr]

    if SWITCH_CALC[0]:

        eig = np.zeros((NUM_ALPHA, SIZE_MAT), dtype=np.complex128)
        mke = np.zeros((NUM_ALPHA, SIZE_MAT), dtype=np.float64)
        mme = np.zeros((NUM_ALPHA, SIZE_MAT), dtype=np.float64)
        ohm = np.zeros((NUM_ALPHA, SIZE_MAT), dtype=np.float64)
        sym = np.full((NUM_ALPHA, SIZE_MAT), str(), dtype=np.str_)

        for i_alpha in range(NUM_ALPHA):
            alpha = LIN_ALPHA[i_alpha]

            logger.info(f'{i_alpha}')

            mat = make_mat(M_ORDER, E_ETA, submatrices, alpha)

            eig_valvec, phys_qtys \
                = solve_eig(M_ORDER, E_ETA, CRITERION_C, alpha, mat)

            eig[i_alpha, :] = eig_valvec[SIZE_MAT, :]
            mke[i_alpha, :] = phys_qtys[0]
            mme[i_alpha, :] = phys_qtys[1]
            ohm[i_alpha, :] = phys_qtys[2]
            sym[i_alpha, :] = phys_qtys[3]

        results = (eig, mke, mme, ohm, sym)

    if SWITCH_CALC[1]:

        eig = np.zeros((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.complex128)
        mke = np.zeros((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.float64)
        mme = np.zeros((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.float64)
        ohm = np.zeros((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.float64)
        sym = np.full((NUM_ALPHA_LOG, SIZE_MAT), str(), dtype=np.str_)

        for i_alpha in range(NUM_ALPHA_LOG):
            alpha = 10**LIN_ALPHA_LOG[i_alpha]

            logger.info(f'{i_alpha}')

            mat = make_mat(M_ORDER, E_ETA, submatrices, alpha)

            eig_valvec, phys_qtys \
                = solve_eig(M_ORDER, E_ETA, CRITERION_C, alpha, mat)

            eig[i_alpha, :] = eig_valvec[SIZE_MAT, :]
            mke[i_alpha, :] = phys_qtys[0]
            mme[i_alpha, :] = phys_qtys[1]
            ohm[i_alpha, :] = phys_qtys[2]
            sym[i_alpha, :] = phys_qtys[3]

        results_log = (eig, mke, mme, ohm, sym)

    return results, results_log


def save_results(results: tuple[ArrayComplex,
                                ArrayFloat,
                                ArrayFloat,
                                ArrayFloat,
                                ArrayStr],
                 results_log: tuple[ArrayComplex,
                                    ArrayFloat,
                                    ArrayFloat,
                                    ArrayFloat,
                                    ArrayStr]) -> None:
    """Save npz files of results.

    Parameters
    ----------
    results : tuple[ArrayComplex, ArrayFloat, ArrayFloat, ArrayFloat,
    ArrayStr]
        The tuple of results (linear-linear).
    results_log : tuple[ArrayComplex, ArrayFloat, ArrayFloat,
    ArrayFloat, ArrayStr]
        The tuple of results (log-log).
    """

    eig: ArrayComplex
    mke: ArrayFloat
    mme: ArrayFloat
    ohm: ArrayFloat
    sym: ArrayStr
    filename: str
    path_file: Path

    os.makedirs(PATH_DIR, exist_ok=True)

    if SWITCH_CALC[0]:

        eig, mke, mme, ohm, sym = results

        filename = NAME_FILE + NAME_FILE_SUFFIX[0]
        path_file = PATH_DIR / filename

        np.savez(path_file,
                 lin_alpha=LIN_ALPHA, eig=eig,
                 mke=mke, mme=mme, ohm=ohm, sym=sym)

    if SWITCH_CALC[1]:

        eig, mke, mme, ohm, sym = results_log

        filename = NAME_FILE + NAME_FILE_SUFFIX[1]
        path_file = PATH_DIR / filename

        np.savez(path_file,
                 lin_alpha=10**LIN_ALPHA_LOG, eig=eig,
                 mke=mke, mme=mme, ohm=ohm, sym=sym)


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    if True not in SWITCH_CALC:
        DefaultLogger(__name__).warning('No saved file')
        sys.exit(0)

    data: tuple[ArrayComplex,
                ArrayFloat,
                ArrayFloat,
                ArrayFloat,
                ArrayStr]
    data_log: tuple[ArrayComplex,
                    ArrayFloat,
                    ArrayFloat,
                    ArrayFloat,
                    ArrayStr]
    data, data_log = wrapper_solve_eig_for_alpha()

    save_results(data, data_log)

    timer.end()
