"""A Python module for the spectral deformation method.

References
----------
[1] John D. Crawford and Peter D. Hislop, Application of the method of
spectral deformation to the Vlasov-Poisson system. Annals of Physics
189, 265-317 (1989).
doi: 10.1016/0003-4916(89)90166-8

[2] John P. Boyd, Chebyshev and Fourier Spectral Methods. Courier
Corporation, (2001).
"""

from package_common.background_field import BackgroundField
from package_common.utils_name import get_current_function_name


def complex_coord_transformation(
        y_start: float | int,
        y_end: float | int,
        *,
        alpha: float | int,
        beta_0: float | int,
        beta_1: float | int) -> BackgroundField:
    """Construct an instance of the BackgroundField class for the
    coordinate transformation, y = y(s), to a complex coordinate in the
    spectral deformation method.

    Parameters
    ----------
    y_start : float
        The starting point.
    y_end : float
        The ending point.
    alpha : float
        A parameter for the transformation to complex coordinates.
    beta_0 : float
        A parameter for the transformation to complex coordinates.
    beta_1 : float
        A parameter for the transformation to complex coordinates.

    Returns
    -------
    BackgroundField
        The instance of the BackgroundField class for the transformation
        to complex coordinates.
    """

    name: str = get_current_function_name()

    def y_complex(s_pos: complex) -> complex:
        return (
            y_start + (y_end-y_start)*(s_pos+1)/2
            - (alpha+1j) * (beta_0+beta_1*s_pos) * ((s_pos**2)-1)
        )

    def y_complex_d(s_pos: complex) -> complex:
        return (
            (y_end-y_start) / 2
            - (alpha+1j) * (beta_1*(3*(s_pos**2)-1)+2*beta_0*s_pos)
        )

    def y_complex_d2(s_pos: complex) -> complex:
        return (
            - 2 * (alpha+1j) * (3*beta_1*s_pos+beta_0)
        )

    return BackgroundField(name,
                           value=y_complex,
                           value_d=y_complex_d,
                           value_d2=y_complex_d2)
