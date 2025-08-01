"""Loads files"""

import logging
import sys
from pathlib import Path

import numpy as np

from package_common.decorator_yesno.py import yes_exe_no_quit
from package_mhd2dsphere.make_legendre import make_legendre

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def wrapper_load_results(switch_plot: tuple[bool, bool],
                         m_order: int,
                         e_eta: float,
                         n_t: int) \
    -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray, np.ndarray],
             tuple[np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, np.ndarray, np.ndarray]]:
    """A wrapper of a function to load .npz files of results

    Parameters
    ----------
    switch_plot : tuple of bool
        Boolean values to switch whether to plot figures
        0: dispersion relation (linear-linear)
        1: dispersion relation (log-log)
    m_order : int
        The zonal wavenumber (order)
    e_eta : float
        The magnetic Ekman number
    n_t : int
        The truncation degree

    Returns
    ----------
    bundle_all : tuple of ndarray
        A tuple of results (linear-linear)
    bundle_all_log : tuple of ndarray
        A tuple of results (log-log)

    """

    name_file: str = f'MHD2Dsphere_sincos_m{m_order}E{e_eta}N{n_t}'
    name_file_suffix: tuple[str, str] = ('.npz', '_log.npz')

    bundle_all: tuple[np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray]
    bundle_all_log: tuple[np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray]

    name_file_full: str
    lin_alpha: np.ndarray
    bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray]

    if switch_plot[0]:

        name_file_full = name_file + name_file_suffix[0]

        lin_alpha, bundle = load_results(name_file_full)

        bundle_all = (lin_alpha, bundle[0], bundle[1],
                      bundle[2], bundle[3], bundle[4])
    else:
        bundle_all = (np.ndarray([]), ) * 6

    if switch_plot[1]:

        name_file_full = name_file + name_file_suffix[1]

        lin_alpha, bundle = load_results(name_file_full)

        lin_alpha = np.log10(lin_alpha)

        bundle_all_log = (lin_alpha, bundle[0], bundle[1],
                          bundle[2], bundle[3], bundle[4])

    return bundle_all, bundle_all_log
#


def load_results(name_file: str) \
    -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]]:
    """Loads .npz files of results

    Parameters
    ----------
    name_file : str
        The name of a loaded file

    Returns
    ----------
    lin_alpha : ndarray
        The sequence of alpha
    bundle : tuple of ndarray
        A tuple of results

    Raises
    ----------
    File not found
        If there are no output files of mhd2dsphere_sincos.py with the
        same parameters.

    """

    path_dir: Path = Path('.') / 'output' / 'MHD2Dsphere_sincos'
    path_file: Path = path_dir / name_file

    lin_alpha: np.ndarray
    eig: np.ndarray
    mke: np.ndarray
    mme: np.ndarray
    ohm: np.ndarray
    sym: np.ndarray

    if not path_file.exists():
        logger.error('File not found')
        sys.exit()
    else:
        npz_kw = np.load(path_file, allow_pickle=True)

        lin_alpha = npz_kw['lin_alpha']
        eig = npz_kw['eig']
        mke = npz_kw['mke']
        mme = npz_kw['mme']
        ohm = npz_kw['ohm']
        sym = npz_kw['sym']

    bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray] \
        = (eig, mke, mme, ohm, sym)

    return lin_alpha, bundle
#


def load_legendre(m_order: int,
                  n_t: int,
                  num_theta: int) -> np.ndarray:
    """Loads data of values of associated Legendre polynomials at grid
    points

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order)
    n_t : int
        The truncation degree
    num_theta : int
        The number of the grid in the theta direction

    Returns
    ----------
    legendre_norm : ndarray
        Values of associated Legendre polynomials at grid points

    Raises
    ----------
    File not found. Do you want to run package/make_legendre.py?
        If there is no output file for the same parameters. Then, you
        can execute package/make_legendre.py.

    """

    @yes_exe_no_quit
    def wrapper_make_legendre(m_order, n_t, num_theta) -> None:
        make_legendre(m_order, n_t, num_theta)

    path_dir: Path = Path('.') / 'output' / 'make_legendre'
    name_file: str = f'make_legendre_m{m_order}N{n_t}th{num_theta}.npy'
    path_file: Path = path_dir / name_file

    legendre_norm: np.ndarray
    if path_file.exists():
        legendre_norm = np.load(path_file)
    else:
        logger.info(
            'File not found. '
            + 'Do you want to run package/make_legendre.py?')
        wrapper_make_legendre(m_order, n_t, num_theta)
        legendre_norm = np.load(path_file)

    return legendre_norm
