"""A Python module to process obtained data from the eigenvalue problem
of two-dimensional (2D) incompressible magnetohydrodynamic (MHD) waves on a
rotating sphere under a toroidal background field, B_phi = B_0 B(theta)
sin(theta), and background zonal flows, U_phi = U_0 U(theta) sin(theta).
"""

import numpy as np

from package_common.common_types import ArrayComplex, ArrayFloat, ArrayStr
from package_mhd2dsphere.typed_dict import DictParams, DictPhysQtys, DictResult


def screening_eig_q(results: DictResult,
                    *,
                    criterion_q: float) -> DictResult:
    """Check the quality factors of eigenmodes.

    Parameters
    ----------
    results : DictResult
        The dictionary of the results of the eigenvalue problem.
    criterion_q : float
        A criterion for plotted eigenvalues based on the quality factor.

    Returns
    -------
    results : DictResult
        The dictionary of the results of the eigenvalue problem.
    """

    lin_alpha: ArrayFloat = results['lin_alpha']
    eig: ArrayComplex = results['eig']
    pke: ArrayFloat = results['phys_qtys']['pke']
    pme: ArrayFloat = results['phys_qtys']['pme']
    psm: ArrayFloat = results['phys_qtys']['psm']
    pse: ArrayFloat = results['phys_qtys']['pse']
    ohm: ArrayFloat = results['phys_qtys']['ohm']
    sym: ArrayStr = results['phys_qtys']['sym']

    size_mat: int = len(eig)

    params: DictParams = pickup_param(results)
    num_alpha: int = params['num_alpha']

    check_q: ArrayFloat
    for i_alpha in range(num_alpha):

        check_q = (
            np.abs(eig[i_alpha, :].real)
            + 2 * eig[i_alpha, :].imag * criterion_q
        )

        for i_mode in range(size_mat):
            if check_q[i_mode] <= 0:
                eig[i_alpha, i_mode] = np.nan
                pke[i_alpha, i_mode] = np.nan
                pme[i_alpha, i_mode] = np.nan
                psm[i_alpha, i_mode] = np.nan
                pse[i_alpha, i_mode] = np.nan
                ohm[i_alpha, i_mode] = np.nan
                sym[i_alpha, i_mode] = np.nan

    phys_qtys: DictPhysQtys = {
        'pke': pke,
        'pme': pme,
        'psm': psm,
        'pse': pse,
        'ohm': ohm,
        'sym': sym
    }

    results = {
        'lin_alpha': lin_alpha,
        'eig': eig,
        'vec_psi': None,
        'vec_vpa': None,
        'phys_qtys': phys_qtys
    }

    return results


def pickup_param(results: DictResult) -> DictParams:
    """Pick up some parameters from results.

    Parameters
    ----------
    results : DictResult
        The dictionary of results.

    Returns
    -------
    params : DictParams
        The dictionary of the parameters.
    """

    lin_alpha: ArrayFloat = results['lin_alpha']
    alpha_init: float = lin_alpha[0]
    alpha_end: float = lin_alpha[-1]
    num_alpha: int = len(lin_alpha)

    ohm_max: float = np.nanmax(results['phys_qtys']['ohm'])

    params: DictParams = {
        'alpha_init': alpha_init,
        'alpha_end': alpha_end,
        'num_alpha': num_alpha,
        'ohm_max': ohm_max
    }

    return params


def pickup_eig(eig: ArrayComplex,
               pke: ArrayFloat,
               sym: ArrayStr) -> dict[str, ArrayComplex]:
    """Pick up the eigenvalues of various modes.

    Parameters
    ----------
    eig : ArrayComplex
        The eigenvalues for a given alpha.
    pke : ArrayFloat
        The perturbation kinetic energies for a given alpha.
    sym : ArrayStr
        The symmetry of the eigenmodes for a given alpha.

    Returns
    -------
    dict_eig : dict[str, ArrayComplex]
        The dictionary to pick up the eigenvalues of various modes.
    """

    sinuous: ArrayFloat
    varicose: ArrayFloat
    sinuous, varicose = sort_sv(sym)

    prograde: ArrayFloat
    retrograde: ArrayFloat
    prograde, retrograde = sort_pr(eig)

    unstable: ArrayFloat = pickup_unstable(eig)

    alfvenic, non_alfvenic = sort_alfvenic(pke)

    dict_eig: dict[str, ArrayComplex] = {
        's': eig * sinuous,
        'v': eig * varicose,

        's_u': eig * sinuous * unstable,
        'v_u': eig * varicose * unstable,

        's_a': eig * sinuous * alfvenic,
        'v_a': eig * varicose * alfvenic,

        's_na': eig * sinuous * non_alfvenic,
        'v_na': eig * varicose * non_alfvenic,

        'sr': eig * sinuous * retrograde,
        'sp': eig * sinuous * prograde,
        'vr': eig * varicose * retrograde,
        'vp': eig * varicose * prograde,

        'sr_u': eig * sinuous * retrograde * unstable,
        'sp_u': eig * sinuous * prograde * unstable,
        'vr_u': eig * varicose * retrograde * unstable,
        'vp_u': eig * varicose * prograde * unstable,

        'sr_a': eig * sinuous * retrograde * alfvenic,
        'sp_a': eig * sinuous * prograde * alfvenic,
        'vr_a': eig * varicose * retrograde * alfvenic,
        'vp_a': eig * varicose * prograde * alfvenic,

        'sr_na': eig * sinuous * retrograde * non_alfvenic,
        'sp_na': eig * sinuous * prograde * non_alfvenic,
        'vr_na': eig * varicose * retrograde * non_alfvenic,
        'vp_na': eig * varicose * prograde * non_alfvenic,
    }

    return dict_eig


def sort_sv(sym: ArrayStr) -> tuple[ArrayFloat,
                                    ArrayFloat]:
    """Sort into sinuous and varicose modes.

    Parameters
    ----------
    sym : ArrayStr
        The symmetry of the eigenmodes for a given alpha.

    Returns
    -------
    sinuous : ArrayFloat
        The identifier of sinuous modes.
    varicose : ArrayFloat
        The identifier of varicose modes.
    """

    size_mat: int = len(sym)

    sinuous: ArrayFloat = np.full(size_mat, np.nan, dtype=np.float64)
    varicose: ArrayFloat = np.full(size_mat, np.nan, dtype=np.float64)
    for i_mode in range(size_mat):
        if sym[i_mode] in ('sinuous', 's'):
            sinuous[i_mode] = 1
        elif sym[i_mode] in ('varicose', 'v'):
            varicose[i_mode] = 1

    return sinuous, varicose


def sort_pr(eig: ArrayComplex) -> tuple[ArrayFloat,
                                        ArrayFloat]:
    """Sort into prograde and retrograde modes.

    Parameters
    ----------
    eig : ArrayComplex
        Eigenvalues for a given alpha

    Returns
    -------
    prograde : ArrayFloat
        The identifier of prograde modes.
    retrograde : ArrayFloat
        The identifier of retrograde modes.
    """

    size_mat: int = len(eig)

    prograde: ArrayFloat = np.full(size_mat, np.nan, dtype=np.float64)
    retrograde: ArrayFloat = np.full(size_mat, np.nan, dtype=np.float64)
    for i_mode in range(size_mat):
        if eig[i_mode].real > 0:
            prograde[i_mode] = 1
        elif eig[i_mode].real < 0:
            retrograde[i_mode] = 1

    return prograde, retrograde


def pickup_unstable(eig: ArrayComplex) -> ArrayFloat:
    """Picks up unstable modes

    Parameters
    ----------
    eig : ArrayComplex
        The eigenvalues for a given alpha.

    Returns
    -------
    unstable : ArrayFloat
        The identifier of unstable modes.
    """

    size_mat: int = len(eig)

    unstable: ArrayFloat = np.full(size_mat, np.nan, dtype=np.float64)
    for i_mode in range(size_mat):
        if np.abs(eig[i_mode].imag) > 0:
            unstable[i_mode] = 1

    return unstable


def sort_alfvenic(pke: ArrayFloat) -> tuple[ArrayFloat,
                                            ArrayFloat]:
    """Sort into alfvenic and non-alfvenic modes.

    Parameters
    ----------
    pke : ArrayFloat
        The perturbation kinetic energy for a given alpha.

    Returns
    ----------
    alfvenic : ArrayFloat
        The identifier of alfvenic modes.
    non_alfvenic : ArrayFloat
        The identifier of non-alfvenic modes
    """

    size_mat: int = len(pke)

    alfvenic: ArrayFloat = np.full(size_mat, np.nan, dtype=np.float64)
    non_alfvenic: ArrayFloat = np.full(size_mat, np.nan, dtype=np.float64)
    for i_mode in range(size_mat):
        if 0.49 < pke[i_mode] < 0.51:
            alfvenic[i_mode] = 1
        else:
            non_alfvenic[i_mode] = 1

    return alfvenic, non_alfvenic
