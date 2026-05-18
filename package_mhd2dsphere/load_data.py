"""A Python module to load the files of the results of the eigenvalue
problem of two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves
on a rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and a background zonal flow, U_phi = U_0 U(theta) sin(theta).
"""

import sys
from pathlib import Path

import numpy as np

from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger
from package_mhd2dsphere.typed_dict import (DictFileInfo, DictPhysQtys,
                                            DictResult)


def wrapper_load_results(switch_plot: tuple[bool, bool, bool],
                         *,
                         info_load: DictFileInfo) \
        -> tuple[DictResult | None,
                 DictResult | None]:
    """Load npz files of the results.

    Parameters
    ----------
    switch_plot : tuple of bool
        The boolean values to switch whether to plot figures or not.
    info_load : DictFileInfo
        The information of the loaded files.

    Returns
    -------
    data : DictResult | None
        The dictionary of the results of the eigenvalue problem
        (linear-linear).
    data_log : DictResult | None
        The dictionary of the results of the eigenvalue problem (log-log).
    """

    data: DictResult | None = None
    data_log: DictResult | None = None

    name_file: str

    if switch_plot[0] or switch_plot[2]:

        name_file = info_load['name_file'] + info_load['name_file_suffix'][0]
        data = load_results(name_file, info_load=info_load)

    if switch_plot[1]:

        name_file = info_load['name_file'] + info_load['name_file_suffix'][1]
        data_log = load_results(name_file, info_load=info_load)

    return data, data_log


def load_results(name_file: str,
                 *,
                 info_load: DictFileInfo) -> DictResult:
    """Load a npz file of the results.

    Parameters
    ----------
    name_file : str
        The name of a loaded file.
    info_load : DictFileInfo
        The information of the loaded files.

    Returns
    -------
    result : DictResult
        The dictionary of the results of the eigenvalue problem.

    Warnings
    --------
    File not found
        If there are no output files of mhd2dsphere_eig.py with the same
        parameters.
    """

    logger: DefaultLogger = create_function_name_logger()
    file_logger: DefaultLogger = DefaultLogger(name_file)

    path_dir: Path = info_load['path_dir']
    path_file: Path = path_dir / name_file

    file_logger.info('Start loading')

    if not path_file.exists():
        logger.error('File not found')
        sys.exit(1)

    npz_kw = np.load(path_file, allow_pickle=True)

    file_logger.info('Loaded')

    phys_qtys: DictPhysQtys = {
        'pke': npz_kw['pke'],
        'pme': npz_kw['pme'],
        'psm': npz_kw['psm'],
        'pse': npz_kw['pse'],
        'ohm': npz_kw['ohm'],
        'sym': npz_kw['sym']
    }

    results: DictResult = {
        'lin_alpha': npz_kw['lin_alpha'],
        'eig': npz_kw['eig'],
        'vec_psi': np.array([]),
        'vec_vpa': np.array([]),
        'phys_qtys': phys_qtys
    }

    return results
