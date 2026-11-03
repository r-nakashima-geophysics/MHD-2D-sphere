"""A Python script to perform time-dependent (quasi-)linear simulations for
two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and a background zonal flow, U_phi = U_0 U(theta) sin(theta).

This script outputs a txt file for information of simulation settings and a npz
file of the simulation results, which include values for the stream function,
vector potential, and background fields.

Parameters
----------
ID : str
    The run index.

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
Run the script with the default value of `ID`.
    $ uv run python mhd2dsphere_linsim.py
Run the script with a specified value of `ID` (say 0001):
    $ uv run python mhd2dsphere_linsim.py 0001
"""

import os
import time
from pathlib import Path

import numpy as np

from package_common.background_field import BackgroundField
from package_common.common_types import Any, ArrayFloat, Final
from package_common.default_logger import DefaultLogger
from package_common.default_timer import DefaultTimer
from package_common.progress_bar import ProgressBar
from package_common.utils_collocation import calc_collocation_point
from package_common.utils_input import input_string
from package_common.utils_name import create_function_name_progress_bar
from package_common.utils_output import output_info
from package_common.utils_simulation import Field, Rhs, time_integrate
from package_mhd2dsphere import (init_background_b, init_background_u,
                                 init_linsim)
from package_mhd2dsphere.create_rhs import wrapper_rhs

# ========== Parameters ========== #

# the run index
ID: Final[str] = input_string(str(int(time.time())))

# The zonal wavenumbers (orders)
M_ORDERS: Final[list[int]] = [2, ]

# The Lehnert number
ALPHA: Final[float] = 0.1

# The magnetic Ekman number
E_ETA: Final[float] = 0

# The Rossby number
ROSSBY: Final[float] = 0

# The truncation degree
N_T: Final[int] = 500

# --- [start] output info --- #
# Time step
NUM_STEP: Final[int] = 10**4
DT: Final[float] = 1e-2
SAVE_INTERVAL: Final[int] = 100

# Initial conditions
INIT_BG_FIELD_B: Final[BackgroundField] = init_background_b.b_malkus('mu')
INIT_BG_FIELD_U: Final[BackgroundField] = init_background_u.u_rigid('mu')
DICT_INIT_PSI: Final[dict[int, ArrayFloat]] = {
    M_ORDERS[0]: init_linsim.init_spherical_harmonics(N_T+1, 2, M_ORDERS[0])
}
DICT_INIT_MVP: Final[dict[int, Any]] = {}

# The boolean value to switch whether to perform quasi-linear simulations or
# not.
SWITCH_QUASI_LIN: Final[bool] = False
# --- [end] output info --- #

# The paths and filenames of outputs
PATH_DIR: Final[Path] = Path('.') / 'output' / 'MHD2Dsphere_linsim'
NAME_FILE: Final[str] = (
    f'MHD2Dsphere_linsim'
    + f'_a={ALPHA}_E={E_ETA}_R={ROSSBY}_N={N_T}_#{ID}'
)
NAME_FILE_SUFFIX: Final[tuple[str, str]] = ('.npz', '.txt')


# ================================ #

SIZE_SUBMAT: Final[int] = N_T + 1
SIZE_MAT: Final[int] = 2 * SIZE_SUBMAT

LINSP_MU: Final[ArrayFloat] = np.array([
    calc_collocation_point(i_l+1, SIZE_SUBMAT+2)
    for i_l in range(SIZE_SUBMAT)])


def initialize_fields(psi: list[Field],
                      mvp: list[Field],
                      bf_u_sin: Field,
                      bf_b_sin: Field) -> tuple[list[Field],
                                                list[Field],
                                                Field,
                                                Field]:
    """Initialize the values of the fields for the (quasi-)linear simulation.

    Parameters
    ----------
    psi : list[Field]
        The list of the stream function (psi).
    mvp : list[Field]
        The list of the vector potential (a).
    bf_u_sin : Field
        The background zonal flow.
    bf_b_sin : Field
        The toroidal background field.

    Returns
    -------
    tuple[list[Field], list[Field], Field, Field]
        The list of fields (list of psi, list of mvp, bf_u_sin, and bf_b_sin).
    """

    for i_m, m_order in enumerate(M_ORDERS):
        if m_order in DICT_INIT_PSI.keys():
            psi[i_m].value += DICT_INIT_PSI[m_order]
        if m_order in DICT_INIT_MVP.keys():
            mvp[i_m].value += DICT_INIT_MVP[m_order]

    bf_u_sin.value += np.array(
        [INIT_BG_FIELD_U.r_value(mu) * np.sqrt(1-mu**2) for mu in LINSP_MU])
    bf_b_sin.value += np.array(
        [INIT_BG_FIELD_B.r_value(mu) * np.sqrt(1-mu**2) for mu in LINSP_MU])

    for field in [*psi, *mvp, bf_u_sin, bf_b_sin]:
        field.store()

    return psi, mvp, bf_u_sin, bf_b_sin


def wrapper_time_integrate(psi: list[Field],
                           mvp: list[Field],
                           bf_u_sin: Field,
                           bf_b_sin: Field) -> tuple[list[Field],
                                                     list[Field],
                                                     Field,
                                                     Field]:
    """Perform time integration for the (quasi-)linear simulation.

    Parameters
    ----------
    psi : list[Field]
        The list of the stream function (psi).
    mvp : list[Field]
        The list of the vector potential (a).
    bf_u_sin : Field
        The background zonal flow.
    bf_b_sin : Field
        The toroidal background field.
    """

    fields: list[Field] = [*psi, *mvp, bf_u_sin, bf_b_sin]
    rhss: list[Rhs] = [
        *[wrapper_rhs('psi', m_order,
                      list_m_order=M_ORDERS, linsp_mu=LINSP_MU,
                      alpha=ALPHA, e_eta=E_ETA, rossby=ROSSBY,
                      switch_quasi_lin=SWITCH_QUASI_LIN)
          for m_order in M_ORDERS],
        *[wrapper_rhs('mvp', m_order,
                      list_m_order=M_ORDERS, linsp_mu=LINSP_MU,
                      alpha=ALPHA, e_eta=E_ETA, rossby=ROSSBY,
                      switch_quasi_lin=SWITCH_QUASI_LIN)
          for m_order in M_ORDERS],
        wrapper_rhs('bf_u_sin', list_m_order=M_ORDERS, linsp_mu=LINSP_MU,
                    alpha=ALPHA, e_eta=E_ETA, rossby=ROSSBY,
                    switch_quasi_lin=SWITCH_QUASI_LIN),
        wrapper_rhs('bf_b_sin', list_m_order=M_ORDERS, linsp_mu=LINSP_MU,
                    alpha=ALPHA, e_eta=E_ETA, rossby=ROSSBY,
                    switch_quasi_lin=SWITCH_QUASI_LIN)
    ]

    progress_bar: ProgressBar \
        = create_function_name_progress_bar(NUM_STEP)
    progress_bar.start()
    for i_step in range(NUM_STEP):

        time_integrate(fields, DT, rhss)

        if (i_step+1) % SAVE_INTERVAL == 0:
            for field in fields:
                field.store()

        progress_bar.update(i_step)

    return psi, mvp, bf_u_sin, bf_b_sin


def save_results(psi: list[Field],
                 mvp: list[Field],
                 bf_u_sin: Field,
                 bf_b_sin: Field) -> None:
    """Save npz files of the simulation results.

    Parameters
    ----------
    psi : list[Field]
        The list of the stream function (psi).
    mvp : list[Field]
        The list of the vector potential (a).
    bf_u_sin : Field
        The background zonal flow.
    bf_b_sin : Field
        The toroidal background field.
    """

    os.makedirs(PATH_DIR, exist_ok=True)

    name_file: str = NAME_FILE + NAME_FILE_SUFFIX[0]
    path_file: Path = PATH_DIR / name_file

    np.savez_compressed(
        path_file,
        m_orders=M_ORDERS,
        linsp_mu=LINSP_MU,
        time=Field.storage_time,
        psi=[field.storage_value for field in psi],
        mvp=[field.storage_value for field in mvp],
        bf_u_sin=bf_u_sin.storage_value,
        bf_b_sin=bf_b_sin.storage_value)

    DefaultLogger(name_file).info('Saved')


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    logger: DefaultLogger = DefaultLogger(__name__)

    logger.show_params(f'{ID=}',
                       f'{M_ORDERS=}',
                       f'{ALPHA=}',
                       f'{E_ETA=}',
                       f'{ROSSBY=}',
                       f'{N_T=}',
                       f'{NUM_STEP=}',
                       f'{DT=}',
                       f'{SWITCH_QUASI_LIN=}')

    path_info: Path = PATH_DIR / (NAME_FILE + NAME_FILE_SUFFIX[1])
    output_info(Path(__file__), path_info)

    Field.set_class_variable(SIZE_SUBMAT)
    psi: list[Field] = [
        Field(f'psi_{m}', dtype=np.complex128) for m in M_ORDERS]
    mvp: list[Field] = [
        Field(f'mvp_{m}', dtype=np.complex128) for m in M_ORDERS]
    bf_u_sin: Field = Field('bf_u_sin')
    bf_b_sin: Field = Field('bf_b_sin')

    psi, mvp, bf_u_sin, bf_b_sin = initialize_fields(
        psi, mvp, bf_u_sin, bf_b_sin)

    psi, mvp, bf_u_sin, bf_b_sin = wrapper_time_integrate(
        psi, mvp, bf_u_sin, bf_b_sin)

    save_results(psi, mvp, bf_u_sin, bf_b_sin)

    timer.end()
