"""A Python module to define a class for handling background fields."""

import numpy as np

from package_common.common_types import Optional, ComplexFunc
from package_common.default_logger import DefaultLogger


class BackgroundField:
    """Class to define background fields.

    Attributes
    ----------
    __logger : DefaultLogger
        The instance of the logger.
    value : ComplexFunc
        The value of the background field at a given point.
    value_d : Optional[ComplexFunc]
        The value of the first derivative of the background field at the
        point.
    value_d2 : Optional[ComplexFunc]
        The value of the second derivative of the background field at
        the point.
    name : str
        The name of the background field.
    tex : str
        The LaTeX text of the background field.

    Warnings
    ----------
    Invalid input type.
        If the input is not a float or int.
    """

    def __init__(self,
                 name: str,
                 value: ComplexFunc,
                 value_d: Optional[ComplexFunc] = None,
                 value_d2: Optional[ComplexFunc] = None,
                 tex: str = '') -> None:
        """Initialize the BackgroundField instance

        Parameters
        ----------
        name : str
            The name of the background field.
        value : Callable[[complex], complex]
            The value of the background field at the point.
        value_d : Callable[[complex], complex], optional, default None
            The value of the first derivative of the background field at
            the point.
        value_d2 : Callable[[complex], complex], optional, default None
            The value of the second derivative of the background field
            at the point.
        tex : str, optional, default ''
            The LaTeX text of the background field.
        """

        self.__logger: DefaultLogger = DefaultLogger(name)

        self.name: str = name
        self.value: ComplexFunc = value
        self.value_d: Optional[ComplexFunc] = value_d
        self.value_d2: Optional[ComplexFunc] = value_d2
        self.tex: str = tex

    def r_value(self,
                x: float | int) -> float:
        """Get the value of the background field at a given (real)
        point.

        Parameters
        ----------
        x: float | int
            The (real) point at which the value of the background field
            is evaluated.

        Returns
        -------
        float
            The value of the background field at the point.
        """

        if not isinstance(x, (float, int)):
            self.__logger.warning('Invalid input type.')

        return self.value(complex(x, 0)).real

    def r_value_d(self,
                  x: float | int) -> float:
        """Get the value of the first derivative of the background field
        at a given (real) point.

        Parameters
        ----------
        x: float | int
            The (real) point at which the value of the first derivative
            of the background field is evaluated.

        Returns
        -------
        float
            The value of the first derivative of the background field at
            the point.
        """

        if not isinstance(x, (float, int)):
            self.__logger.warning('Invalid input type.')

        return self.value_d(complex(x, 0)).real

    def r_value_d2(self,
                   x: float | int) -> float:
        """Get the value of the second derivative of the background
        field at a given (real) point.

        Parameters
        ----------
        x: float | int
            The (real) point at which the value of the second derivative
            of the background field is evaluated.

        Returns
        -------
        float
            The value of the second derivative of the background field
            at the point.
        """

        if not isinstance(x, (float, int)):
            self.__logger.warning('Invalid input type.')

        return self.value_d2(complex(x, 0)).real
