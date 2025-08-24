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
from package_common.spectral_deform import (ComplexCoordinate,
                                            init_complex_coordinate)
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
BG_FIELD_B: Final[BackgroundField] = init_background_b.b_sincos('mu')
BG_FIELD_U: Final[BackgroundField] = init_background_u.u_rigid('mu')
# For the spectral deformation method
COMPLEX_MU: Final[ComplexCoordinate] = init_complex_coordinate(
    -1, 1, alpha=0, beta_0=0, beta_1=0)
# The boolean value to switch whether to follow Nakashima & Yoshida
# (2024)[1]_ or not
# If SWITCH_NY24 is True, BG_FIELD_B, BG_FIELD_U and
# COMPLEX_MU are ignored.
SWITCH_NY24: Final[bool] = False

# The zonal wavenumber (order)
M_ORDER: Final[int] = input_value(1, int)

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The Rossby number
ROSSBY: Final[float] = 0

# The truncation degree
N_T: Final[int] = 100
# N_T: Final[int] = 2000

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
    + f'_m{M_ORDER}E{E_ETA}R{ROSSBY}N{N_T}' \
    + f'{COMPLEX_MU.name}'
NAME_FILE_SUFFIX: Final[tuple[str, str]] = ('.npz', '_log.npz')

# The number of processes for multiprocessing
NUM_PROCESS: Final[int] = set_num_process()
# The number of threads for each process
NUM_THREADS: Final[int] = 1

# ================================ #

BG_FIELD: Final[DictBackgroundField] = {
    'B': BG_FIELD_B,
    'U': BG_FIELD_U,
    'MU': COMPLEX_MU,
    'NY24': SWITCH_NY24,
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

size_submat: int
if SWITCH_NY24:
    size_submat = N_T - M_ORDER + 1
else:
    size_submat = N_T + 1
SIZE_SUBMAT: Final[int] = size_submat
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT


def wrapper_solve_eig_for_alpha(*,
                                switch_log: bool = False) \
    -> tuple[ArrayComplex,
             ArrayFloat,
             ArrayFloat,
             ArrayFloat,
             ArrayStr]:
    """Solve the eigenvalue problem for given sequences of alpha.

    Parameters
    ----------
    switch_log : bool, optional, default False
        The boolean value for the dispersion problem of the log-log
        plot.

    Returns
    -------
    eig : ArrayComplex
        The eigenvalues.
    mke : ArrayFloat
        The mean kinetic energy.
    mme : ArrayFloat
        The mean magnetic energy.
    ohm : ArrayFloat
        The ohmic dissipation.
    sym : ArrayStr
        The symmetry of eigenmodes.
    """

    submatrices: tuple[ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex] \
        = create_submat(M_ORDER, E_ETA, ROSSBY, SIZE_SUBMAT,
                        background_field=BG_FIELD)

    shared_memories: tuple[SharedMemory, ...]
    shared_info: list[tuple[str, tuple[int, ...], np.dtype]]
    shared_memories, shared_info = create_shared_arrays(*submatrices)

    num_alpha: int
    args_list: list[tuple[float,
                          list[tuple[str, tuple[int, ...], np.dtype]]]]
    if not switch_log:
        num_alpha = NUM_ALPHA
        args_list = [(LIN_ALPHA[i_alpha], shared_info)
                     for i_alpha in range(NUM_ALPHA)]
    else:
        num_alpha = NUM_ALPHA_LOG
        args_list = [(10**LIN_ALPHA_LOG[i_alpha], shared_info)
                     for i_alpha in range(NUM_ALPHA_LOG)]

    eig: ArrayComplex \
        = np.empty((num_alpha, SIZE_MAT), dtype=np.complex128)
    mke: ArrayFloat = np.empty((num_alpha, SIZE_MAT), dtype=np.float64)
    mme: ArrayFloat = np.empty((num_alpha, SIZE_MAT), dtype=np.float64)
    ohm: ArrayFloat = np.empty((num_alpha, SIZE_MAT), dtype=np.float64)
    sym: ArrayStr = np.empty((num_alpha, SIZE_MAT), dtype=np.str_)

    progress_bar: ProgressBar \
        = create_function_name_progress_bar(num_alpha)
    progress_bar.start()
    with multiprocessing.Pool(processes=NUM_PROCESS,
                              initializer=set_num_threads,
                              initargs=(NUM_THREADS,)) as pool:
        for i_alpha, result in enumerate(pool.imap(worker, args_list)):

            eig[i_alpha, :] = result[0][SIZE_MAT, :]
            mke[i_alpha, :] = result[1][0]
            mme[i_alpha, :] = result[1][1]
            ohm[i_alpha, :] = result[1][2]
            sym[i_alpha, :] = result[1][3]

            progress_bar.update(i_alpha, NUM_PROCESS)

    detach_shared_arrays(*shared_memories, unlink=True)

    return eig, mke, mme, ohm, sym


def worker(args: tuple[float,
                       list[tuple[str, tuple[int, ...], np.dtype]]]) \
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
                                ArrayStr],
                 *,
                 switch_log: bool = False) -> None:
    """Save npz files of the results.

    Parameters
    ----------
    results : tuple[ArrayComplex, ArrayFloat, ArrayFloat, ArrayFloat,
    ArrayStr]
        The tuple of the results (linear-linear).
    switch_log : bool, optional, default False
        The boolean value for the dispersion problem of the log-log
        plot.
    """

    eig: ArrayComplex
    mke: ArrayFloat
    mme: ArrayFloat
    ohm: ArrayFloat
    sym: ArrayStr

    os.makedirs(PATH_DIR, exist_ok=True)

    eig, mke, mme, ohm, sym = results

    name_file: str
    if not switch_log:
        name_file = NAME_FILE + NAME_FILE_SUFFIX[0]
        lin_alpha = LIN_ALPHA
    else:
        name_file = NAME_FILE + NAME_FILE_SUFFIX[1]
        lin_alpha = 10**LIN_ALPHA_LOG
    path_file: Path = PATH_DIR / name_file

    np.savez_compressed(path_file,
                        lin_alpha=lin_alpha, eig=eig,
                        mke=mke, mme=mme, ohm=ohm, sym=sym)


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    if not any(SWITCH_CALC):
        DefaultLogger(__name__).warning('No saved file')
        sys.exit(0)

    data: tuple[ArrayComplex,
                ArrayFloat,
                ArrayFloat,
                ArrayFloat,
                ArrayStr] | None = None

    if SWITCH_CALC[0]:
        data = wrapper_solve_eig_for_alpha()
        save_results(data)
    if SWITCH_CALC[1]:
        data = wrapper_solve_eig_for_alpha(switch_log=True)
        save_results(data, switch_log=True)

    timer.end()
