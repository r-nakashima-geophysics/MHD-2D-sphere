"""A Python script to calculate the dispersion relation of
two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta).

This script outputs up to two npz files of results, which include alpha,
eigenvalue, perturbation kinetic energy, perturbation magnetic energy,
pseudomomentum, pseudoenergy, ohmic dissipation, and the symmetry of
eigenmodes. In addition, the script uses multiprocessing to speed up the
calculations.

Parameters
----------
M_ORDER : int
    The zonal wavenumber (order).

Warnings
--------
No saved file
    If all of the boolean values to switch whether to calculate or not are
    False.
Invalid settings
    If E_ETA != 0 and the spectral deformation method is used.

Notes
-----
All other parameters aside from command line arguments are described within the
script.

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
                                         Final, SharedInfo, SharedMemory)
from package_common.default_logger import DefaultLogger
from package_common.default_timer import DefaultTimer
from package_common.progress_bar import ProgressBar
from package_common.spectral_deform import (ComplexCoordinate,
                                            init_complex_coordinate_simple)
from package_common.utils_input import input_value
from package_common.utils_name import create_function_name_progress_bar
from package_common.utils_parallel import (attach_shared_arrays,
                                           create_shared_arrays,
                                           detach_shared_arrays,
                                           set_num_process, set_num_threads)
from package_mhd2dsphere import init_background_b, init_background_u
from package_mhd2dsphere.create_mat import create_mat, create_submat
from package_mhd2dsphere.solve_eig import solve_eig
from package_mhd2dsphere.typed_dict import (DictBackgroundField,
                                            DictCriterionC, DictPhysQtys,
                                            DictResult)

# ========== Parameters ========== #

# The boolean values to switch whether to calculate or not
# SWITCH_CALC[0]: The dispersion relation for the linear-linear plot
# SWITCH_CALC[1]: The dispersion relation for the log-log plot
SWITCH_CALC: Final[tuple[bool, bool]] = (True, True)

# Background field
BG_FIELD_B: Final[BackgroundField] = init_background_b.b_malkus('mu')
BG_FIELD_U: Final[BackgroundField] = init_background_u.u_rigid('mu')
# For the spectral deformation method
MU_COMPLEX: Final[ComplexCoordinate] = init_complex_coordinate_simple(
    -1, 1, alpha=0, beta_0=0, beta_1=0)
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
    = f'MHD2Dsphere_eig_NY24_m={M_ORDER}_E={E_ETA}_N={N_T}' \
    if SWITCH_NY24 \
    else f'MHD2Dsphere_eig_B{BG_FIELD_B.name}U{BG_FIELD_U.name}' \
    + f'_m={M_ORDER}_E={E_ETA}_R={ROSSBY}_N={N_T}' \
    + f'_{MU_COMPLEX.name}'
NAME_FILE_SUFFIX: Final[tuple[str, str]] = ('.npz', '_log.npz')

# The number of processes for multiprocessing
NUM_PROCESS: Final[int] = set_num_process()
# The number of threads for each process
NUM_THREADS: Final[int] = 1

# ================================ #

BG_FIELD: Final[DictBackgroundField] = {
    'B': BG_FIELD_B,
    'U': BG_FIELD_U,
    'MU': MU_COMPLEX,
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

SIZE_SUBMAT: Final[int] = N_T - M_ORDER + 1 if SWITCH_NY24 else N_T + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT


def wrapper_solve_eig_for_lin_alpha(*, switch_log: bool = False) -> DictResult:
    """Solve the eigenvalue problem for given sequences of alpha.

    Parameters
    ----------
    switch_log : bool, optional, default False
        The boolean value for the dispersion problem of the log-log
        plot.

    Returns
    -------
    results : DictResult
        The dictionary of the results of the eigenvalue problem.
    """

    submatrices: tuple[ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex,
                       ArrayFloat | ArrayComplex] \
        = create_submat(M_ORDER, E_ETA, ROSSBY, SIZE_SUBMAT,
                        background_field=BG_FIELD)

    shared_memories: tuple[SharedMemory, ...]
    shared_info: SharedInfo
    shared_memories, shared_info = create_shared_arrays(*submatrices)

    try:
        num_alpha: int
        args_list: list[tuple[float, SharedInfo]]
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
        pke: ArrayFloat = np.empty((num_alpha, SIZE_MAT), dtype=np.float64)
        pme: ArrayFloat = np.empty((num_alpha, SIZE_MAT), dtype=np.float64)
        psm: ArrayFloat = np.empty((num_alpha, SIZE_MAT), dtype=np.float64)
        pse: ArrayFloat = np.empty((num_alpha, SIZE_MAT), dtype=np.float64)
        ohm: ArrayFloat = np.empty((num_alpha, SIZE_MAT), dtype=np.float64)
        sym: ArrayStr = np.empty((num_alpha, SIZE_MAT), dtype=np.str_)

        progress_bar: ProgressBar \
            = create_function_name_progress_bar(num_alpha)
        progress_bar.start()
        with multiprocessing.Pool(processes=NUM_PROCESS,
                                  initializer=set_num_threads,
                                  initargs=(NUM_THREADS,)) as pool:
            for i_alpha, result in enumerate(pool.imap(worker, args_list)):

                eig[i_alpha, :] = result['eig']
                pke[i_alpha, :] = result['phys_qtys']['pke']
                pme[i_alpha, :] = result['phys_qtys']['pme']
                psm[i_alpha, :] = result['phys_qtys']['psm']
                pse[i_alpha, :] = result['phys_qtys']['pse']
                ohm[i_alpha, :] = result['phys_qtys']['ohm']
                sym[i_alpha, :] = result['phys_qtys']['sym']

                progress_bar.update(i_alpha, NUM_PROCESS)

    finally:
        detach_shared_arrays(*shared_memories, unlink=True)

    phys_qtys: DictPhysQtys = {
        'pke': pke,
        'pme': pme,
        'psm': psm,
        'pse': pse,
        'ohm': ohm,
        'sym': sym
    }

    results: DictResult = {
        'lin_alpha': LIN_ALPHA if not switch_log else 10**LIN_ALPHA_LOG,
        'eig': eig,
        'vec_psi': None,
        'vec_vpa': None,
        'phys_qtys': phys_qtys
    }

    return results


def worker(args: tuple[float, SharedInfo]) -> DictResult:
    """Set the task for multiprocessing.

    Parameters
    ----------
    args : tuple[float, SharedInfo]
        The arguments for the task.

    Returns
    -------
    DictResult
        The results of the task.
    """

    alpha: float
    shared_info: SharedInfo
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


def save_results(results: DictResult,
                 *,
                 switch_log: bool = False) -> None:
    """Save npz files of the results.

    Parameters
    ----------
    results : DictResult
        The dictionary of the results of the eigenvalue problem.
    switch_log : bool, optional, default False
        The boolean value for the dispersion problem of the log-log
        plot.
    """

    lin_alpha: ArrayFloat = results['lin_alpha']
    eig: ArrayComplex = results['eig']
    pke: ArrayFloat = results['phys_qtys']['pke']
    pme: ArrayFloat = results['phys_qtys']['pme']
    psm: ArrayFloat = results['phys_qtys']['psm']
    pse: ArrayFloat = results['phys_qtys']['pse']
    ohm: ArrayFloat = results['phys_qtys']['ohm']
    sym: ArrayStr = results['phys_qtys']['sym']

    os.makedirs(PATH_DIR, exist_ok=True)

    name_file: str
    if not switch_log:
        name_file = NAME_FILE + NAME_FILE_SUFFIX[0]
    else:
        name_file = NAME_FILE + NAME_FILE_SUFFIX[1]
    path_file: Path = PATH_DIR / name_file

    np.savez_compressed(path_file,
                        lin_alpha=lin_alpha, eig=eig,
                        pke=pke, pme=pme, psm=psm, pse=pse,
                        ohm=ohm, sym=sym)

    DefaultLogger(name_file).info('Saved')


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    logger: DefaultLogger = DefaultLogger(__name__)

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

    if not any(SWITCH_CALC):
        logger.warning('No saved file')
        sys.exit(0)

    if (E_ETA != 0) and MU_COMPLEX.check_spectral_deform():
        logger.warning('Invalid settings')
        sys.exit(1)

    data: DictResult

    if SWITCH_CALC[0]:
        data = wrapper_solve_eig_for_lin_alpha()
        save_results(data)
    if SWITCH_CALC[1]:
        data = wrapper_solve_eig_for_lin_alpha(switch_log=True)
        save_results(data, switch_log=True)

    timer.end()
