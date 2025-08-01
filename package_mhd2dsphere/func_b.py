"""Defines the function B(theta) for the non-Malkus field
B_phi = B_0 B(theta) sin(theta)

"""

import cmath
from typing import Callable


def b_hydro(switch_theta: str = 'mu') \
    -> tuple[Callable[[complex], complex], Callable[[complex], complex],
             Callable[[complex], complex], str, str]:
    """The hydrodynamic case (B=0)

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The argument to switch whether you use either mu or theta

    Returns
    ----------
    hydro : callable
        A function to calculate the value of B
    hydro_d : callable
        A function to calculate the value of the first derivative of B
    hydro_d2 : callable
        A function to calculate the value of the second derivative of B
    tex : str
        TeX text
    name : str
        Name

    """

    def hydro_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    def hydro_d_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    def hydro_d2_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    hydro: Callable[[complex], complex] = hydro_mu
    hydro_d: Callable[[complex], complex] = hydro_d_mu
    hydro_d2: Callable[[complex], complex] = hydro_d2_mu

    if switch_theta == 'theta':
        def hydro_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def hydro_d_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def hydro_d2_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        hydro = hydro_theta
        hydro_d = hydro_d_theta
        hydro_d2 = hydro_d2_theta

    tex: str = r'0'
    name: str = 'hydro'

    return hydro, hydro_d, hydro_d2, tex, name
#


def b_malkus(switch_theta: str = 'mu') \
    -> tuple[Callable[[complex], complex], Callable[[complex], complex],
             Callable[[complex], complex], str, str]:
    """The Malkus field (B=1)

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The argument to switch whether you use either mu or theta

    Returns
    ----------
    malkus : callable
        A function to calculate the value of B
    malkus_d : callable
        A function to calculate the value of the first derivative of B
    malkus_d2 : callable
        A function to calculate the value of the second derivative of B
    tex : str
        TeX text
    name : str
        Name

    """

    def malkus_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 1

    def malkus_d_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    def malkus_d2_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    malkus: Callable[[complex], complex] = malkus_mu
    malkus_d: Callable[[complex], complex] = malkus_d_mu
    malkus_d2: Callable[[complex], complex] = malkus_d2_mu

    if switch_theta == 'theta':
        def malkus_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 1

        def malkus_d_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        def malkus_d2_theta(theta_complex: complex) -> complex:
            _ = theta_complex
            return 0

        malkus = malkus_theta
        malkus_d = malkus_d_theta
        malkus_d2 = malkus_d2_theta

    tex: str = r'B_0\sin\theta'
    name: str = 'malkus'

    return malkus, malkus_d, malkus_d2, tex, name
#


def b_sincos(switch_theta: str = 'mu') \
    -> tuple[Callable[[complex], complex], Callable[[complex], complex],
             Callable[[complex], complex], str, str]:
    """B = cos(theta)

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The argument to switch whether you use either mu or theta

    Returns
    ----------
    sincos : callable
        A function to calculate the value of B
    sincps_d : callable
        A function to calculate the value of the first derivative of B
    sincos_d2 : callable
        A function to calculate the value of the second derivative of B
    tex : str
        TeX text
    name : str
        Name

    """

    def sincos_mu(mu_complex: complex) -> complex:
        return mu_complex

    def sincos_d_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 1

    def sincos_d2_mu(mu_complex: complex) -> complex:
        _ = mu_complex
        return 0

    sincos: Callable[[complex], complex] = sincos_mu
    sincos_d: Callable[[complex], complex] = sincos_d_mu
    sincos_d2: Callable[[complex], complex] = sincos_d2_mu

    if switch_theta == 'theta':
        def sincos_theta(theta_complex: complex) -> complex:
            return cmath.cos(theta_complex)

        def sincos_d_theta(theta_complex: complex) -> complex:
            return -cmath.sin(theta_complex)

        def sincos_d2_theta(theta_complex: complex) -> complex:
            return -cmath.cos(theta_complex)

        sincos = sincos_theta
        sincos_d = sincos_d_theta
        sincos_d2 = sincos_d2_theta

    tex: str = r'B_0\sin\theta\cos\theta'
    name: str = 'sincos'

    return sincos, sincos_d, sincos_d2, tex, name
#


def b_sin2cos(switch_theta: str = 'mu') \
    -> tuple[Callable[[complex], complex], Callable[[complex], complex],
             Callable[[complex], complex], str, str]:
    """B = sin(theta) cos(theta)

    Parameters
    ----------
    switch_theta : str, default 'mu'
        The argument to switch whether you use either mu or theta

    Returns
    ----------
    sin2cos : callable
        A function to calculate the value of B
    sincps_d : callable
        A function to calculate the value of the first derivative of B
    sin2cos_d2 : callable
        A function to calculate the value of the second derivative of B
    tex : str
        TeX text
    name : str
        Name

    """

    def sin2cos_mu(mu_complex: complex) -> complex:
        return mu_complex * cmath.sqrt(1-(mu_complex**2))

    def sin2cos_d_mu(mu_complex: complex) -> complex:
        return (1-2*(mu_complex**2)) / cmath.sqrt(1-(mu_complex**2))

    def sin2cos_d2_mu(mu_complex: complex) -> complex:
        return mu_complex * (2*(mu_complex**2)-3) \
            / (cmath.sqrt(1-(mu_complex**2))**3)

    sin2cos: Callable[[complex], complex] = sin2cos_mu
    sin2cos_d: Callable[[complex], complex] = sin2cos_d_mu
    sin2cos_d2: Callable[[complex], complex] = sin2cos_d2_mu

    if switch_theta == 'theta':
        def sin2cos_theta(theta_complex: complex) -> complex:
            return cmath.sin(theta_complex) * cmath.cos(theta_complex)

        def sin2cos_d_theta(theta_complex: complex) -> complex:
            return cmath.cos(2*theta_complex)

        def sin2cos_d2_theta(theta_complex: complex) -> complex:
            return -2 * cmath.sin(2*theta_complex)

        sin2cos = sin2cos_theta
        sin2cos_d = sin2cos_d_theta
        sin2cos_d2 = sin2cos_d2_theta

    tex: str = r'B_0\sin^2\theta\cos\theta'
    name: str = 'sin2cos'

    return sin2cos, sin2cos_d, sin2cos_d2, tex, name
#
