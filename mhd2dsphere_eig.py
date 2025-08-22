"""A Python script to calculate the dispersion relation of
two-dimensional (2D) magnetohydrodynamic (MHD) waves on a rotating
sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta).

This script outputs up to two npz files of results, which include alpha,
eigenvalue, mean kinetic energy, mean magnetic energy, ohmic
dissipation, and the symmetry of eigenmodes. In addition, the script
uses multiprocessing to speed up the calculations.

Parameters
----------
M_ORDER : int
    The zonal wavenumber (order).

Warnings
--------
No saved file
    If all of the boolean values to switch whether to calculate or not
    are False.

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

[2] Ryosuke Nakashima (in prep.)

Examples
--------
Run the script with the default value of M_ORDER:
    $ python3 mhd2dsphere_eig.py
Run the script with a specified value (say M_ORDER = 2):
    $ python3 mhd2dsphere_eig.py 2
"""

import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

from package_common.background_field import BackgroundField
from package_common.common_types import (ArrayComplex, ArrayFloat, ArrayStr,
                                         Final, SharedMemory)
from package_common.default_logger import DefaultLogger
from package_common.default_timer import DefaultTimer
from package_common.progress_bar import ProgressBar
from package_common.utils_input import input_value
from package_common.utils_name import create_function_name_progress_bar
from package_common.utils_parallel import (attach_shared_arrays,
                                           create_shared_arrays,
                                           detach_shared_arrays,
                                           set_num_process, set_num_threads)
from package_mhd2dsphere import init_background_b, init_background_u
from package_mhd2dsphere.create_mat import create_mat, create_submat
from package_mhd2dsphere.solve_eig import solve_eig
from package_mhd2dsphere.typed_dict import DictBackgroundField, DictCriterionC

# ========== Parameters ========== #

# The boolean values to switch whether to calculate or not
# SWITCH_CALC[0]: The dispersion relation for the linear-linear plot
# SWITCH_CALC[1]: The dispersion relation for the log-log plot
SWITCH_CALC: Final[tuple[bool, bool]] = (True, True)

# Background field
BG_FIELD_B: Final[BackgroundField] = init_background_b.b_hydro('mu')
BG_FIELD_U: Final[BackgroundField] = init_background_u.u_rigid('mu')
# The boolean value to switch whether to follow Nakashima & Yoshida
# (2024)[1]_ or not
# If SWITCH_NY24 is True, BG_FIELD_B and BG_FIELD_U are ignored.
SWITCH_NY24: Final[bool] = True

# The zonal wavenumber (order)
M_ORDER: Final[int] = input_value(1, int)

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The truncation degree
N_T: Final[int] = 500

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
PATH_DIR: Final[Path] = Path('.') / 'output' / 'MHD2Dsphere_eig'
NAME_FILE: Final[str] \
    = f'MHD2Dsphere_eig_NY24_m{M_ORDER}E{E_ETA}N{N_T}' \
    if SWITCH_NY24 \
    else f'MHD2Dsphere_eig_B{BG_FIELD_B.name}U{BG_FIELD_U.name}' \
    + f'_m{M_ORDER}E{E_ETA}N{N_T}'
NAME_FILE_SUFFIX: Final[tuple[str, str]] = ('.npz', '_log.npz')

# The number of processes for multiprocessing
NUM_PROCESS: Final[int] = set_num_process()
# The number of threads for each process
NUM_THREADS: Final[int] = 1

# ================================ #

BG_FIELD: Final[DictBackgroundField] = {
    'B': BG_FIELD_B,
    'U': BG_FIELD_U,
    'NY24': SWITCH_NY24
}

CRITERION_C: Final[DictCriterionC] = {
    'degree': N_C,
    'ratio': R_C
}

NUM_ALPHA: Final[int] = 1 + int((ALPHA_END-ALPHA_INIT)/ALPHA_STEP)
NUM_ALPHA_LOG: Final[int] \
    = 1 + int((ALPHA_LOG_END-ALPHA_LOG_INIT)/ALPHA_LOG_STEP)

LIN_ALPHA: Final[ArrayFloat] = np.linspace(
    ALPHA_INIT, ALPHA_END, NUM_ALPHA, dtype=np.float64)
LIN_ALPHA_LOG: Final[ArrayFloat] = np.linspace(
    ALPHA_LOG_INIT, ALPHA_LOG_END, NUM_ALPHA_LOG, dtype=np.float64)

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT


def wrapper_solve_eig_for_alpha() -> tuple[tuple[ArrayComplex,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayStr] | None,
                                           tuple[ArrayComplex,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayFloat,
                                                 ArrayStr] | None]:
    """Solve the eigenvalue problem for given sequences of alpha.

    Returns
    -------
    results : tuple[ArrayComplex, ArrayFloat, ArrayFloat, ArrayFloat,
    ArrayStr] | None
        The tuple of results (linear-linear).
    results_log : tuple[ArrayComplex, ArrayFloat, ArrayFloat,
    ArrayFloat, ArrayStr] | None
        The tuple of results (log-log).
    """

    submatrices: tuple[ArrayFloat,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayFloat] \
        = create_submat(M_ORDER, SIZE_SUBMAT,
                        background_field=BG_FIELD)

    shared_memories: tuple[SharedMemory, ...]
    shared_info: list[tuple[str, tuple[int, ...], np.dtype]]
    shared_memories, shared_info = create_shared_arrays(*submatrices)

    results: tuple[ArrayComplex,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayStr] | None = None
    results_log: tuple[ArrayComplex,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayStr] | None = None

    eig: ArrayComplex
    mke: ArrayFloat
    mme: ArrayFloat
    ohm: ArrayFloat
    sym: ArrayStr

    progress_bar: ProgressBar

    if SWITCH_CALC[0]:

        eig = np.empty((NUM_ALPHA, SIZE_MAT), dtype=np.complex128)
        mke = np.empty((NUM_ALPHA, SIZE_MAT), dtype=np.float64)
        mme = np.empty((NUM_ALPHA, SIZE_MAT), dtype=np.float64)
        ohm = np.empty((NUM_ALPHA, SIZE_MAT), dtype=np.float64)
        sym = np.empty((NUM_ALPHA, SIZE_MAT), dtype=np.str_)

        args_list = [(LIN_ALPHA[i_alpha], shared_info)
                     for i_alpha in range(NUM_ALPHA)]

        progress_bar = create_function_name_progress_bar(NUM_ALPHA)
        progress_bar.start()
        with multiprocessing.Pool(processes=NUM_PROCESS,
                                  initializer=set_num_threads,
                                  initargs=(NUM_THREADS,)) as pool:
            for i_alpha, result in enumerate(
                    pool.imap(worker, args_list)):

                eig[i_alpha, :] = result[0][SIZE_MAT, :]
                mke[i_alpha, :] = result[1][0]
                mme[i_alpha, :] = result[1][1]
                ohm[i_alpha, :] = result[1][2]
                sym[i_alpha, :] = result[1][3]

                progress_bar.update(i_alpha, NUM_PROCESS)

        results = (eig, mke, mme, ohm, sym)

    if SWITCH_CALC[1]:

        eig = np.empty((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.complex128)
        mke = np.empty((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.float64)
        mme = np.empty((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.float64)
        ohm = np.empty((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.float64)
        sym = np.empty((NUM_ALPHA_LOG, SIZE_MAT), dtype=np.str_)

        args_list = [(10**LIN_ALPHA_LOG[i_alpha], shared_info)
                     for i_alpha in range(NUM_ALPHA_LOG)]

        progress_bar = create_function_name_progress_bar(NUM_ALPHA_LOG)
        progress_bar.start()
        with multiprocessing.Pool(processes=NUM_PROCESS,
                                  initializer=set_num_threads,
                                  initargs=(NUM_THREADS,)) as pool:
            for i_alpha, result in enumerate(
                    pool.imap(worker, args_list)):

                eig[i_alpha, :] = result[0][SIZE_MAT, :]
                mke[i_alpha, :] = result[1][0]
                mme[i_alpha, :] = result[1][1]
                ohm[i_alpha, :] = result[1][2]
                sym[i_alpha, :] = result[1][3]

                progress_bar.update(i_alpha, NUM_PROCESS)

        results_log = (eig, mke, mme, ohm, sym)

    detach_shared_arrays(*shared_memories, unlink=True)

    return results, results_log


def worker(args: tuple[float,
                       list[tuple[str,
                                  tuple[int, ...],
                                  np.dtype]]]) \
    -> tuple[ArrayComplex,
             tuple[ArrayFloat,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayStr]]:
    """Set the task for multiprocessing.

    Parameters
    ----------
    args : tuple[float, list[tuple[str, tuple[int, ...], np.dtype]]]
        The arguments for the task.

    Returns
    -------
    tuple[ArrayComplex, tuple[ArrayFloat, ArrayFloat, ArrayFloat,
    ArrayStr]]
        The results of the task.
    """

    alpha: float
    shared_info: list[tuple[str,
                            tuple[int, ...],
                            np.dtype]]
    alpha, shared_info = args

    shared_memories: tuple[SharedMemory,
                           SharedMemory,
                           SharedMemory,
                           SharedMemory]
    submatrices: tuple[ArrayFloat,
                       ArrayFloat,
                       ArrayFloat,
                       ArrayFloat]
    shared_memories, submatrices = attach_shared_arrays(shared_info)

    mat = create_mat(M_ORDER, alpha, E_ETA, submatrices,
                     background_field=BG_FIELD)

    detach_shared_arrays(*shared_memories)

    return solve_eig(M_ORDER, alpha, E_ETA, mat,
                     criterion_c=CRITERION_C,
                     background_field=BG_FIELD)


def save_results(results: tuple[ArrayComplex,
                                ArrayFloat,
                                ArrayFloat,
                                ArrayFloat,
                                ArrayStr] | None,
                 results_log: tuple[ArrayComplex,
                                    ArrayFloat,
                                    ArrayFloat,
                                    ArrayFloat,
                                    ArrayStr] | None) -> None:
    """Save npz files of the results.

    Parameters
    ----------
    results : tuple[ArrayComplex, ArrayFloat, ArrayFloat, ArrayFloat,
    ArrayStr] | None
        The tuple of the results (linear-linear).
    results_log : tuple[ArrayComplex, ArrayFloat, ArrayFloat,
    ArrayFloat, ArrayStr] | None
        The tuple of the results (log-log).
    """

    eig: ArrayComplex
    mke: ArrayFloat
    mme: ArrayFloat
    ohm: ArrayFloat
    sym: ArrayStr
    filename: str
    path_file: Path

    os.makedirs(PATH_DIR, exist_ok=True)

    if results is not None:

        eig, mke, mme, ohm, sym = results

        filename = NAME_FILE + NAME_FILE_SUFFIX[0]
        path_file = PATH_DIR / filename

        np.savez(path_file,
                 lin_alpha=LIN_ALPHA, eig=eig,
                 mke=mke, mme=mme, ohm=ohm, sym=sym)

    if results_log is not None:

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
                ArrayStr] | None
    data_log: tuple[ArrayComplex,
                    ArrayFloat,
                    ArrayFloat,
                    ArrayFloat,
                    ArrayStr] | None
    data, data_log = wrapper_solve_eig_for_alpha()

    save_results(data, data_log)

    timer.end()
