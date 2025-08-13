"""A Python module to generate the instances of BackgroundField class
for toroidal background field, B_phi = B_0 B(theta) sin(theta).

Examples
--------
>>> b_hydro = init_b_hydro('mu')
>>> b_hydro = init_b_hydro('theta')
"""

import cmath
import inspect
import sys

from package_common.background_field import BackgroundField
from package_common.default_logger import DefaultLogger


def init_b_hydro(switch_theta: str = 'mu') -> BackgroundField:
    """Generate the instance of BackgroundField class for the
    hydrodynamic case (B=0).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta
        is used.

    Returns
    ----------
    BackgroundField
        The instance of BackgroundField class for the hydrodynamic
        case (B=0).

    Warnings
    ----------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    name: str = 'hydro'
    tex: str = r'0'

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
    
    logger.warning('Invalid argument')
    sys.exit(1)


def init_b_malkus(switch_theta: str = 'mu') -> BackgroundField:
    """Generate the instance of BackgroundField class for the
    Malkus field (B=1).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta
        is used.

    Returns
    ----------
    BackgroundField
        The instance of BackgroundField class for the Malkus field
        (B=1).

    Warnings
    ----------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    name: str = 'malkus'
    tex: str = r'B_0\sin\theta'

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

    logger.warning('Invalid argument')
    sys.exit(1)


def init_b_sincos(switch_theta: str = 'mu') -> BackgroundField:
    """Generate the instance of BackgroundField class for B =
    cos(theta).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta
        is used.

    Returns
    ----------
    BackgroundField
        The instance of BackgroundField class for B = cos(theta).

    Warnings
    ----------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    name: str = 'sincos'
    tex: str = r'B_0\sin\theta\cos\theta'

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
            return cmath.cos(theta_complex)

        def b_sincos_d_theta(theta_complex: complex) -> complex:
            return -cmath.sin(theta_complex)

        def b_sincos_d2_theta(theta_complex: complex) -> complex:
            return -cmath.cos(theta_complex)

        return BackgroundField(name,
                               value=b_sincos_theta,
                               value_d=b_sincos_d_theta,
                               value_d2=b_sincos_d2_theta,
                               tex=tex)

    logger.warning('Invalid argument')
    sys.exit(1)


def init_b_sin2cos(switch_theta: str = 'mu') -> BackgroundField:
    """Generate the instance of BackgroundField class for B = sin(theta)
    cos(theta).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta
        is used.

    Returns
    ----------
    BackgroundField
        The instance of BackgroundField class for B = sin(theta)
        cos(theta).

    Warnings
    ----------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    name: str = 'sin2cos'
    tex: str = r'B_0\sin^2\theta\cos\theta'

    if switch_theta == 'mu':
        def b_sin2cos_mu(mu_complex: complex) -> complex:
            return mu_complex * cmath.sqrt(1-(mu_complex**2))

        def b_sin2cos_d_mu(mu_complex: complex) -> complex:
            return (1-2*(mu_complex**2)) / cmath.sqrt(1-(mu_complex**2))

        def b_sin2cos_d2_mu(mu_complex: complex) -> complex:
            return mu_complex * (2*(mu_complex**2)-3) \
                / (cmath.sqrt(1-(mu_complex**2))**3)

        return BackgroundField(name,
                               value=b_sin2cos_mu,
                               value_d=b_sin2cos_d_mu,
                               value_d2=b_sin2cos_d2_mu,
                               tex=tex)

    if switch_theta == 'theta':
        def b_sin2cos_theta(theta_complex: complex) -> complex:
            return cmath.sin(theta_complex) * cmath.cos(theta_complex)

        def b_sin2cos_d_theta(theta_complex: complex) -> complex:
            return cmath.cos(2*theta_complex)

        def b_sin2cos_d2_theta(theta_complex: complex) -> complex:
            return -2 * cmath.sin(2*theta_complex)

        return BackgroundField(name,
                               value=b_sin2cos_theta,
                               value_d=b_sin2cos_d_theta,
                               value_d2=b_sin2cos_d2_theta,
                               tex=tex)

    logger.warning('Invalid argument')
    sys.exit(1)
