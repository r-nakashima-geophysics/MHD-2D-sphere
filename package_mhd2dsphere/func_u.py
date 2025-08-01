"""Defines the function U(theta) for background flows 
U_phi = U_0 U(theta) sin(theta)

"""

import cmath
from typing import Callable


def u_rigid(switch_theta: str = 'mu') \
    -> tuple[Callable[[complex], complex], Callable[[complex], complex],
             Callable[[complex], complex], str, str]:
    """A rigid body rotation (U=0)

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The argument to switch whether you use either mu or theta

    Returns
    ----------
    rigid : callable
        A function to calculate the value of U
    rigid_d : callable
        A function to calculate the value of the first derivative of U
    rigid_d2 : callable
        A function to calculate the value of the second derivative of U
    tex : str
        TeX text
    name : str
        Name

    """

    def rigid_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    def rigid_d_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    def rigid_d2_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    rigid: Callable[[complex], complex] = rigid_mu
    rigid_d: Callable[[complex], complex] = rigid_d_mu
    rigid_d2: Callable[[complex], complex] = rigid_d2_mu

    if switch_theta == 'theta':
        def rigid_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def rigid_d_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def rigid_d2_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        rigid = rigid_theta
        rigid_d = rigid_d_theta
        rigid_d2 = rigid_d2_theta

    tex: str = r'0'
    name: str = 'rigid'

    return rigid, rigid_d, rigid_d2, tex, name
#
