"""Makes data of the associated Legendre polynomials"""

import math
import os
from pathlib import Path
from time import perf_counter

import caffeine
import numpy as np
from scipy.special import lpmv


def make_legendre(m_order: int,
                  n_t: int,
                  num_theta: int) -> None:
    """Makes data of values of associated Legendre polynomials at grid
    points

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order)
    n_t : int
        The truncation degree
    num_theta : int
        The number of the grid in the theta direction

    """

    time_init: float = perf_counter()

    caffeine.on(display=False)

    path_dir: Path = Path('.') / 'output' / 'make_legendre'
    name_file: str = f'make_legendre_m{m_order}N{n_t}th{num_theta}.npy'
    path_file: Path = path_dir / name_file

    lin_theta: np.ndarray = np.linspace(0, math.pi, num_theta)

    factorial_part: float
    norm: float
    legendre_norm: np.ndarray = np.zeros((n_t+1, num_theta))
    for n_order in range(n_t+1):
        if n_order >= m_order:

            factorial_part = math.factorial(n_order-m_order) \
                / math.factorial(n_order+m_order)
            norm = ((-1)**m_order) \
                * math.sqrt(((2*n_order+1)/2)*factorial_part)

            legendre_norm[n_order, :] \
                = norm * lpmv(m_order, n_order, np.cos(lin_theta))

    os.makedirs(path_dir, exist_ok=True)

    np.save(path_file, legendre_norm)

    time_elapsed: float = perf_counter() - time_init
    print(f'{__name__}: {time_elapsed:.3f} s')


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

    logger: DefaultLogger = create_function_name_logger()

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
