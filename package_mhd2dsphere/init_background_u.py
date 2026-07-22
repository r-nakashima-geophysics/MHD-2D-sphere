"""A Python module to construct the instances of BackgroundField class
for background zonal flows, U_phi = U_0 U(theta) sin(theta)."""

from package_common.background_field import BackgroundField
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


def u_rigid(switch_theta: str = 'mu') -> BackgroundField:
    """Construct an instance of the BackgroundField class for the rigid
    body rotation (U=0).

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The string to switch whether either mu (= cos(theta)) or theta is used.

    Returns
    -------
    BackgroundField
        The instance of the BackgroundField class for the rigid body rotation
        (U=0).

    Warnings
    --------
    Invalid argument
        If the argument is neither 'mu' nor 'theta'.

    Examples
    --------
    >>> from package_mhd2dsphere import init_background_u
    >>> u_rigid = init_background_u.u_rigid('mu')
    >>> u_rigid = init_background_u.u_rigid('theta')
    """

    logger: DefaultLogger = create_function_name_logger()

    name: str = 'rigid'
    tex: str = r'$U_{0\phi}=0$'

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

    logger.error('Invalid argument')
