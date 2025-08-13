"""A Python module to generate the instances of BackgroundField class
for background zonal flows, U_phi = U_0 U(theta) sin(theta).

Example
-------
>>> u_rigid = init_u_rigid('mu')
>>> u_rigid = init_u_rigid('theta')
"""

import inspect
import sys

from package_common.background_field import BackgroundField
from package_common.default_logger import DefaultLogger


def init_u_rigid(switch_theta: str = 'mu') -> BackgroundField:
    """Generate the instance of BackgroundField class for the rigid body
    rotation (U=0).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta
        is used.

    Returns
    ----------
    BackgroundField
        The instance of BackgroundField class for the rigid body
        rotation (U=0).

    Warnings
    ----------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.
    """

    function_name: str = inspect.currentframe().f_code.co_name
    logger: DefaultLogger = DefaultLogger(function_name)

    name: str = 'rigid'
    tex: str = r'0'

    if switch_theta == 'mu':
        def u_rigid_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        def u_rigid_d_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        def u_rigid_d2_mu(mu_complex: complex) -> complex:
            _ = mu_complex
            return 0

        return BackgroundField(name, 
                               value=u_rigid_mu, 
                               value_d=u_rigid_d_mu,
                               value_d2=u_rigid_d2_mu,
                               tex=tex)

    if switch_theta == 'theta':
        def u_rigid_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def u_rigid_d_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def u_rigid_d2_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        return BackgroundField(name, 
                               value=u_rigid_theta, 
                               value_d=u_rigid_d_theta,
                               value_d2=u_rigid_d2_theta,
                               tex=tex)

    logger.warning('Invalid argument')
    sys.exit(1)