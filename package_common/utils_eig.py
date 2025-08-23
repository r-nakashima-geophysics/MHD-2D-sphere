"""A Python module to provide the utilities for eigenvalue problems."""

import sys

import numpy as np

from package_common.common_types import ArrayAny, ArrayBool, ArrayComplex
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


def sort_eig(eigenvalues: ArrayComplex,
             eigenvectors: ArrayComplex) -> ArrayComplex:
    """Sort the eigenvalues and eigenvectors.

    Parameters
    ----------
    eigenvalues : ArrayComplex
        The eigenvalues.
    eigenvectors : ArrayComplex
        The eigenvectors.

    Returns
    -------
    matrix_eig : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.

    Examples
    --------
    >>> from package_common.utils_eig import sort_eig
    >>> eigenvalues, eigenvectors = np.linalg.eig(matrix)
    >>> matrix_eig = sort_eig(eigenvalues, eigenvectors)
    """

    size_matrix: int = len(eigenvalues)

    tmp: ArrayComplex = np.empty(
        (2*size_matrix+2, size_matrix), dtype=np.complex128)
    tmp[0*size_matrix:1*size_matrix, :] = eigenvectors.real
    tmp[1*size_matrix:2*size_matrix, :] = eigenvectors.imag
    tmp[2*size_matrix, :] = eigenvalues.real
    tmp[2*size_matrix+1, :] = eigenvalues.imag
    tmp_sorted: list[ArrayComplex] \
        = sorted(tmp.T, key=lambda x: x[2*size_matrix])
    tmp = np.array(tmp_sorted).T

    matrix_eig: ArrayComplex = np.empty(
        (size_matrix+1, size_matrix), dtype=np.complex128)
    matrix_eig[0*size_matrix:1*size_matrix, :] \
        = tmp[0*size_matrix:1*size_matrix, :] \
        + 1j*tmp[1*size_matrix:2*size_matrix, :]
    matrix_eig[size_matrix, :] \
        = tmp[2*size_matrix, :] + 1j*tmp[2*size_matrix+1, :]

    return matrix_eig


def screening_eig(matrix_eig: ArrayComplex,
                  check: ArrayBool,
                  *phys_qtys) -> tuple[ArrayComplex,
                                       tuple[ArrayAny, ...]]:
    """Exclude invalid eigenmodes.

    Parameters
    ----------
    matrix_eig : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    check : ArrayBool
        The validity of the eigenmodes.
    *phys_qtys
        The physical quantities.

    Returns
    -------
    matrix_eig : ArrayComplex
        The eigenvalues and normalized eigenvectors.
    phys_qtys : tuple[ArrayAny, ...]
        The physical quantities.

    Warnings
    --------
    Invalid shape of the input arrays
        If the shapes of the input arrays do not match.

    Examples
    --------
    >>> from package_common.utils_eig import sort_eig, screening_eig
    >>> eigenvalues, eigenvectors = np.linalg.eig(matrix)
    >>> matrix_eig = sort_eig(eigenvalues, eigenvectors)
    >>> matrix_eig, phys_qtys = screening_eig(matrix_eig, check,
    phys_qtys)
    """

    logger: DefaultLogger = create_function_name_logger()

    size_mat: int = matrix_eig.shape[1]

    if (matrix_eig.shape[0] != size_mat + 1) \
            or (len(check.ravel()) != size_mat):
        logger.error('Invalid shape of the input arrays')
        sys.exit(1)

    for phys_qty in phys_qtys:
        if len(phys_qty.ravel()) != size_mat:
            logger.error('Invalid shape of the input arrays')
            sys.exit(1)

    list_phys_qtys: list[ArrayAny] = list(phys_qtys)

    for i_mode in range(size_mat):
        if not check[i_mode]:
            matrix_eig[:, i_mode] = np.nan

            for phys_qty in list_phys_qtys:
                phys_qty[i_mode] = np.nan

    phys_qtys = tuple(list_phys_qtys)

    return matrix_eig, phys_qtys
