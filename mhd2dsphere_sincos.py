"""A code for 2D MHD waves on a rotating sphere under the non-Malkus
field B_phi = B_0 sin(theta) cos(theta)

Outputs .npz files of results (alpha, eigenvalue, mean kinetic energy,
mean magnetic energy, ohmic dissipation, and symmetry of eigenmodes)
concerning the dispersion relation for 2D MHD waves on a rotating
sphere under the non-Malkus field B_phi = B_0 sin(theta) cos(theta).

Parameters
----------
M_ORDER : int
    The zonal wavenumber (order)

Raises
----------
No saved file
    If all of the boolean values to switch whether to calculate are
    False.

Notes
----------
Parameters other than command line arguments are described below.

References
----------
[1] Nakashima & Yoshida (submitted)

Examples
----------
In the below example, M_ORDER will be set to the default value.
    python3 mhd2dsphere_sincos.py
In the below example, M_ORDER will be set to 2.
    python3 mhd2dsphere_sincos.py 2

"""

import logging
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Final

import caffeine
import numpy as np

from package_common.input_arg import input_int
from package_mhd2dsphere.make_mat import make_mat, make_submat
from package_mhd2dsphere.solve_eig import solve_eig
from package_mhd2dsphere.time_measure import time_progress

# ========== Parameters ==========

# Boolean values to switch whether to calculate
# 0: dispersion relation (linear-linear)
# 1: dispersion relation (log-log)
SWITCH_CALC: Final[tuple[bool, bool]] = (True, True)

# The zonal wavenumber (order)
M_ORDER: Final[int] = input_int(1)

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The truncation degree
N_T: Final[int] = 2000

# A criterion for convergence
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

CRITERION_C: Final[tuple[int, float]] = (N_C, R_C)

NUM_ALPHA: Final[int] \
    = 1 + int((ALPHA_END-ALPHA_INIT)/ALPHA_STEP)
NUM_ALPHA_LOG: Final[int] \
    = 1 + int((ALPHA_LOG_END-ALPHA_LOG_INIT)/ALPHA_LOG_STEP)

LIN_ALPHA: Final[np.ndarray] \
    = np.linspace(ALPHA_INIT, ALPHA_END, NUM_ALPHA)
LIN_ALPHA_LOG: Final[np.ndarray] \
    = np.linspace(ALPHA_LOG_INIT, ALPHA_LOG_END, NUM_ALPHA_LOG)

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT


def wrapper_solve_eig_foralpha() \
    -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray],
             tuple[np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray]]:
    """A wrapper of a function to solve the eigenvalue problem

    Returns
    ----------
    bundle : tuple of ndarray
        A tuple of results (linear-linear)
    bundle_log : tuple of ndarray
        A tuple of results (log-log)


    """

    submatrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] \
        = make_submat(M_ORDER, SIZE_SUBMAT)

    bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray] = (np.array([]), ) * 5
    bundle_log: tuple[np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray] = (np.array([]), ) * 5

    now: float = perf_counter()

    eig: np.ndarray
    mke: np.ndarray
    mme: np.ndarray
    ohm: np.ndarray
    sym: np.ndarray

    alpha: float
    mat: np.ndarray
    eig_vecval: np.ndarray
    phys_qtys: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    if SWITCH_CALC[0]:

        eig = np.zeros((NUM_ALPHA, SIZE_MAT), dtype=np.complex128)
        mke = np.zeros((NUM_ALPHA, SIZE_MAT))
        mme = np.zeros((NUM_ALPHA, SIZE_MAT))
        ohm = np.zeros((NUM_ALPHA, SIZE_MAT))
        sym = np.full((NUM_ALPHA, SIZE_MAT), str(), dtype=object)

        for i_alpha in range(NUM_ALPHA):
            alpha = LIN_ALPHA[i_alpha]

            mat = make_mat(M_ORDER, E_ETA, submatrices, alpha)

            eig_vecval, phys_qtys \
                = solve_eig(M_ORDER, E_ETA, CRITERION_C, alpha, mat)

            eig[i_alpha, :] = eig_vecval[SIZE_MAT, :]
            mke[i_alpha, :] = phys_qtys[0]
            mme[i_alpha, :] = phys_qtys[1]
            ohm[i_alpha, :] = phys_qtys[2]
            sym[i_alpha, :] = phys_qtys[3]

            now = time_progress(NUM_ALPHA, i_alpha, now)
        #

        bundle = (eig, mke, mme, ohm, sym)
    #

    if SWITCH_CALC[1]:

        eig = np.zeros((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.complex128)
        mke = np.zeros((NUM_ALPHA_LOG, SIZE_MAT))
        mme = np.zeros((NUM_ALPHA_LOG, SIZE_MAT))
        ohm = np.zeros((NUM_ALPHA_LOG, SIZE_MAT))
        sym = np.full((NUM_ALPHA, SIZE_MAT), str(), dtype=object)

        for i_alpha in range(NUM_ALPHA_LOG):
            alpha = 10**LIN_ALPHA_LOG[i_alpha]

            mat = make_mat(M_ORDER, E_ETA, submatrices, alpha)

            eig_vecval, phys_qtys \
                = solve_eig(M_ORDER, E_ETA, CRITERION_C, alpha, mat)

            eig[i_alpha, :] = eig_vecval[SIZE_MAT, :]
            mke[i_alpha, :] = phys_qtys[0]
            mme[i_alpha, :] = phys_qtys[1]
            ohm[i_alpha, :] = phys_qtys[2]
            sym[i_alpha, :] = phys_qtys[3]

            now = time_progress(NUM_ALPHA_LOG, i_alpha, now)
        #

        bundle_log = (eig, mke, mme, ohm, sym)
    #

    return bundle, bundle_log
#


def save_results(bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray],
                 bundle_log: tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]) -> None:
    """Saves files

    Parameters
    ----------
    bundle : tuple of ndarray
        A tuple of results (linear-linear)
    bundle_log : tuple of ndarray
        A tuple of results (log-log)

    """

    eig: np.ndarray
    mke: np.ndarray
    mme: np.ndarray
    ohm: np.ndarray
    sym: np.ndarray
    name_file_full: str
    path_file: Path

    if SWITCH_CALC[0]:

        eig, mke, mme, ohm, sym = bundle

        name_file_full = NAME_FILE + NAME_FILE_SUFFIX[0]
        path_file = PATH_DIR / name_file_full

        os.makedirs(PATH_DIR, exist_ok=True)

        np.savez(path_file,
                 lin_alpha=LIN_ALPHA, eig=eig, mke=mke,
                 mme=mme, ohm=ohm, sym=sym)
    #

    if SWITCH_CALC[1]:

        eig, mke, mme, ohm, sym = bundle_log

        name_file_full = NAME_FILE + NAME_FILE_SUFFIX[1]
        path_file = PATH_DIR / name_file_full

        os.makedirs(PATH_DIR, exist_ok=True)

        np.savez(path_file,
                 lin_alpha=10**LIN_ALPHA_LOG, eig=eig, mke=mke,
                 mme=mme, ohm=ohm, sym=sym)
    #
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    logging.basicConfig(level=logging.INFO)
    logger: logging.Logger = logging.getLogger(__name__)

    if True not in SWITCH_CALC:
        logger.info('No saved file')
        sys.exit()
    #

    caffeine.on(display=False)

    results: tuple[np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray]
    results_log: tuple[np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray, np.ndarray]
    results, results_log = wrapper_solve_eig_foralpha()

    save_results(results, results_log)

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')
#
