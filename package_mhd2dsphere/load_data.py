"""A Python module to load the files of the results of the eigenvalue
problem of two-dimensional (2D) magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and background zonal flows, U_phi = U_0 U(theta)
sin(theta)."""

import sys
from pathlib import Path

import numpy as np

from package_common.common_types import ArrayComplex, ArrayFloat, ArrayStr
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger
from package_mhd2dsphere.typed_dict import DictFileInfo


def wrapper_load_results(switch_plot: tuple[bool, bool, bool],
                         *,
                         info_load: DictFileInfo) \
    -> tuple[tuple[ArrayFloat,
                   ArrayComplex,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayStr] | None,
             tuple[ArrayFloat,
                   ArrayComplex,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayFloat,
                   ArrayStr] | None]:
    """Load npz files of the results.

    Parameters
    ----------
    info_load : DictFileInfo
        The information of the loaded files.
    switch_plot : tuple of bool
        The boolean values to switch whether to plot figures or not.

    Returns
    -------
    data : tuple[ArrayFloat, ArrayComplex, ArrayFloat, ArrayFloat,
    ArrayFloat, ArrayStr] | None
        The tuple of results (linear-linear).
    data_log : tuple[ArrayFloat, ArrayComplex, ArrayFloat, ArrayFloat,
    ArrayFloat, ArrayStr] | None
        The tuple of results (log-log).
    """

    data: tuple[ArrayFloat,
                ArrayComplex,
                ArrayFloat,
                ArrayFloat,
                ArrayFloat,
                ArrayStr] | None = None
    data_log: tuple[ArrayFloat,
                    ArrayComplex,
                    ArrayFloat,
                    ArrayFloat,
                    ArrayFloat,
                    ArrayStr] | None = None

    name_file: str

    if switch_plot[0] or switch_plot[2]:

        name_file \
            = info_load['name_file'] + info_load['name_file_suffix'][0]

        data = load_results(name_file, info_load=info_load)

    if switch_plot[1]:

        name_file \
            = info_load['name_file'] + info_load['name_file_suffix'][1]

        tmp_tuple: tuple[ArrayFloat,
                         ArrayComplex,
                         ArrayFloat,
                         ArrayFloat,
                         ArrayFloat,
                         ArrayStr] \
            = load_results(name_file, info_load=info_load)

        lin_alpha = np.log10(tmp_tuple[0])

        data_log = (lin_alpha, tmp_tuple[1], tmp_tuple[2],
                    tmp_tuple[3], tmp_tuple[4], tmp_tuple[5])

    return data, data_log


def load_results(name_file: str,
                 *,
                 info_load: DictFileInfo) -> tuple[ArrayFloat,
                                                   ArrayComplex,
                                                   ArrayFloat,
                                                   ArrayFloat,
                                                   ArrayFloat,
                                                   ArrayStr]:
    """Load a npz file of the results.

    Parameters
    ----------
    name_file : str
        The name of a loaded file.
    info_load : DictFileInfo
        The information of the loaded files.

    Returns
    -------
    lin_alpha : ArrayFloat
        The sequence of alpha.
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

    Warnings
    --------
    File not found
        If there are no output files of mhd2dsphere_eig.py with the
        same parameters.
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

    lin_alpha: ArrayFloat = npz_kw['lin_alpha']
    eig: ArrayComplex = npz_kw['eig']
    mke: ArrayFloat = npz_kw['mke']
    mme: ArrayFloat = npz_kw['mme']
    ohm: ArrayFloat = npz_kw['ohm']
    sym: ArrayStr = npz_kw['sym']

    file_logger.info('Loaded')

    return lin_alpha, eig, mke, mme, ohm, sym
