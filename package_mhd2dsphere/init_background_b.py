"""A Python module to construct the instances of BackgroundField class
for toroidal background fields, B_phi = B_0 B(theta) sin(theta)."""

import numpy as np

from package_common.background_field import BackgroundField
from package_common.common_types import NoReturn
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


def is_symmetric_b(b_field: BackgroundField) -> bool | NoReturn:
    """Check whether the toroidal background field is equatorial
    (anti)symmetric or not.

    Parameters
    ----------
    b_field : BackgroundField
        The instance of the BackgroundField class.

    Returns
    -------
    bool
        The boolean value to check whether the background field is equatorial
        (anti)symmetric or not.

    Warnings
    --------
    Unknown background field name
        If the name of the background field is undefined.
    """

    logger: DefaultLogger = create_function_name_logger()

    if b_field.name == 'hydro':
        return True

    if b_field.name == 'malkus':
        return True

    if b_field.name == 'sincos':
        return True

    if b_field.name == 'sin2cos':
        return True

    if b_field.name.startswith('malsincos'):
        return False

    logger.error('Unknown background field name')


def b_hydro(switch_theta: str = 'mu') -> BackgroundField | NoReturn:
    """Construct an instance of the BackgroundField class for the
    hydrodynamic case (B=0).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta is used.

    Returns
    -------
    BackgroundField
        The instance of the BackgroundField class for the hydrodynamic case
        (B=0).

    Warnings
    --------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.

    Examples
    --------
    >>> from package_mhd2dsphere import init_background_b
    >>> b_hydro = init_background_b.b_hydro('mu')
    >>> b_hydro = init_background_b.b_hydro('theta')
    """

    logger: DefaultLogger = create_function_name_logger()

    name: str = 'hydro'
    tex: str = r'$0$'

    if switch_theta == 'mu':
        def b_hydro_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        def b_hydro_d_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        def b_hydro_d2_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        return BackgroundField(name,
                               value=b_hydro_mu,
                               value_d=b_hydro_d_mu,
                               value_d2=b_hydro_d2_mu,
                               tex=tex)

    if switch_theta == 'theta':
        def b_hydro_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def b_hydro_d_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def b_hydro_d2_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        return BackgroundField(name,
                               value=b_hydro_theta,
                               value_d=b_hydro_d_theta,
                               value_d2=b_hydro_d2_theta,
                               tex=tex)

    logger.error('Invalid argument')


def b_malkus(switch_theta: str = 'mu') -> BackgroundField | NoReturn:
    """Construct an instance of the BackgroundField class for the
    Malkus field (B=1).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta is used.

    Returns
    -------
    BackgroundField
        The instance of the BackgroundField class for the Malkus field (B=1).

    Warnings
    --------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.

    Examples
    --------
    >>> from package_mhd2dsphere import init_background_b
    >>> b_malkus = init_background_b.b_malkus('mu')
    >>> b_malkus = init_background_b.b_malkus('theta')
    """

    logger: DefaultLogger = create_function_name_logger()

    name: str = 'malkus'
    tex: str = r'$B_{0\phi}=B_0\sin\theta$'

    if switch_theta == 'mu':
        def b_malkus_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 1

        def b_malkus_d_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        def b_malkus_d2_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        return BackgroundField(name,
                               value=b_malkus_mu,
                               value_d=b_malkus_d_mu,
                               value_d2=b_malkus_d2_mu,
                               tex=tex)

    if switch_theta == 'theta':
        def b_malkus_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 1

        def b_malkus_d_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def b_malkus_d2_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        return BackgroundField(name,
                               value=b_malkus_theta,
                               value_d=b_malkus_d_theta,
                               value_d2=b_malkus_d2_theta,
                               tex=tex)

    logger.error('Invalid argument')


def b_sincos(switch_theta: str = 'mu') -> BackgroundField | NoReturn:
    """Construct an instance of the BackgroundField class for B =
    cos(theta).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta is used.

    Returns
    -------
    BackgroundField
        The instance of the BackgroundField class for B = cos(theta).

    Warnings
    --------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.

    Examples
    --------
    >>> from package_mhd2dsphere import init_background_b
    >>> b_sincos = init_background_b.b_sincos('mu')
    >>> b_sincos = init_background_b.b_sincos('theta')
    """

    logger: DefaultLogger = create_function_name_logger()

    name: str = 'sincos'
    tex: str = r'$B_{0\phi}=B_0\sin\theta\cos\theta$'

    if switch_theta == 'mu':
        def b_sincos_mu(mu_complex: complex) -> complex:
            return mu_complex

        def b_sincos_d_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 1

        def b_sincos_d2_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        return BackgroundField(name,
                               value=b_sincos_mu,
                               value_d=b_sincos_d_mu,
                               value_d2=b_sincos_d2_mu,
                               tex=tex)

    if switch_theta == 'theta':
        def b_sincos_theta(theta_complex: complex) -> complex:
            return np.cos(theta_complex)

        def b_sincos_d_theta(theta_complex: complex) -> complex:
            return -np.sin(theta_complex)

        def b_sincos_d2_theta(theta_complex: complex) -> complex:
            return -np.cos(theta_complex)

        return BackgroundField(name,
                               value=b_sincos_theta,
                               value_d=b_sincos_d_theta,
                               value_d2=b_sincos_d2_theta,
                               tex=tex)

    logger.error('Invalid argument')


def b_sin2cos(switch_theta: str = 'mu') -> BackgroundField | NoReturn:
    """Construct an instance of the BackgroundField class for B =
    sin(theta) cos(theta).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta is used.

    Returns
    -------
    BackgroundField
        The instance of the BackgroundField class for B = sin(theta)
        cos(theta).

    Warnings
    --------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.

    Examples
    --------
    >>> from package_mhd2dsphere import init_background_b
    >>> b_sin2cos = init_background_b.b_sin2cos('mu')
    >>> b_sin2cos = init_background_b.b_sin2cos('theta')
    """

    logger: DefaultLogger = create_function_name_logger()

    name: str = 'sin2cos'
    tex: str = r'$B_{0\phi}=B_0\sin^2\theta\cos\theta$'

    if switch_theta == 'mu':
        def b_sin2cos_mu(mu_complex: complex) -> complex:
            return mu_complex * np.sqrt(1-(mu_complex**2))

        def b_sin2cos_d_mu(mu_complex: complex) -> complex:
            return (1-2*(mu_complex**2)) / np.sqrt(1-(mu_complex**2))

        def b_sin2cos_d2_mu(mu_complex: complex) -> complex:
            return mu_complex * (2*(mu_complex**2)-3) \
                / (np.sqrt(1-(mu_complex**2))**3)

        return BackgroundField(name,
                               value=b_sin2cos_mu,
                               value_d=b_sin2cos_d_mu,
                               value_d2=b_sin2cos_d2_mu,
                               tex=tex)

    if switch_theta == 'theta':
        def b_sin2cos_theta(theta_complex: complex) -> complex:
            return np.sin(theta_complex) * np.cos(theta_complex)

        def b_sin2cos_d_theta(theta_complex: complex) -> complex:
            return np.cos(2*theta_complex)

        def b_sin2cos_d2_theta(theta_complex: complex) -> complex:
            return -2 * np.sin(2*theta_complex)

        return BackgroundField(name,
                               value=b_sin2cos_theta,
                               value_d=b_sin2cos_d_theta,
                               value_d2=b_sin2cos_d2_theta,
                               tex=tex)

    logger.error('Invalid argument')


def b_malsincos(ratio_sincos2malkus: float,
                switch_theta: str = 'mu') -> BackgroundField | NoReturn:
    """Construct an instance of the BackgroundField class for B =
    const * (1 + ratio_sincos2malkus * cos(theta)).

    Parameters
    ----------
    ratio_sincos2malkus : float
        The ratio of the sincos part to the Malkus part of the background
        field.
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta is used.

    Returns
    -------
    BackgroundField
        The instance of the BackgroundField class for B = const * (1 +
        ratio_sincos2malkus * cos(theta)).

    Warnings
    --------
    Use b_malkus() or b_sincos() instead of b_malsincos()
        If the ratio_sincos2malkus is 0 or infinity.
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.

    Examples
    --------
    >>> from package_mhd2dsphere import init_background_b
    >>> b_malsincos = init_background_b.b_malsincos(1.0, 'mu')
    >>> b_malsincos = init_background_b.b_malsincos(1.0, 'theta')
    """

    logger: DefaultLogger = create_function_name_logger()

    if (ratio_sincos2malkus == 0) or (ratio_sincos2malkus == np.inf):
        logger.error('Use b_malkus() or b_sincos() instead of b_malsincos()')

    name: str = f'malsincos{ratio_sincos2malkus:.2f}'

    malkus: float = 1 / (1+ratio_sincos2malkus)
    sincos: float = ratio_sincos2malkus / (1+ratio_sincos2malkus)

    tex: str = r'$B_{0\phi}=B_0\sin\theta(' \
        + f'{malkus:.2f}' + r' + ' + f'{sincos:.2f}' + r'\cos\theta)$'

    if switch_theta == 'mu':
        def b_malsincos_mu(mu_complex: complex) -> complex:
            return malkus + sincos * mu_complex

        def b_malsincos_d_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return sincos

        def b_malsincos_d2_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        return BackgroundField(name,
                               value=b_malsincos_mu,
                               value_d=b_malsincos_d_mu,
                               value_d2=b_malsincos_d2_mu,
                               tex=tex)

    if switch_theta == 'theta':
        def b_malsincos_theta(theta_complex: complex) -> complex:
            return malkus + sincos * np.cos(theta_complex)

        def b_malsincos_d_theta(theta_complex: complex) -> complex:
            return -sincos * np.sin(theta_complex)

        def b_malsincos_d2_theta(theta_complex: complex) -> complex:
            return -sincos * np.cos(theta_complex)

        return BackgroundField(name,
                               value=b_malsincos_theta,
                               value_d=b_malsincos_d_theta,
                               value_d2=b_malsincos_d2_theta,
                               tex=tex)

    logger.error('Invalid argument')
