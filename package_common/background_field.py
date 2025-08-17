"""A Python module to define a class for handling background fields."""

import sys

from package_common.common_types import ComplexFunc, Optional
from package_common.default_logger import DefaultLogger


class BackgroundField:
    """Class to define background fields.

    Attributes
    ----------
    __logger : DefaultLogger
        The instance of the logger.
    value : ComplexFunc
        The value of the background field at a given point.
    __value_d : Optional[ComplexFunc]
        The value of the first derivative of the background field at the
        point.
    __value_d2 : Optional[ComplexFunc]
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
    Attribute has not been set.
        If the attribute is not set.

    Example
    -------
    >>> linear = BackgroundField('linear', lambda x: x)
    >>> linear.value(1)
    1
    """

    def __init__(self,
                 name: str,
                 value: ComplexFunc,
                 value_d: Optional[ComplexFunc] = None,
                 value_d2: Optional[ComplexFunc] = None,
                 tex: Optional[str] = None) -> None:
        """Initialize the BackgroundField instance

        Parameters
        ----------
        name : str
            The name of the background field.
        value : ComplexFunc
            The value of the background field at the point.
        value_d : Optional[ComplexFunc], optional, default None
            The value of the first derivative of the background field at
            the point.
        value_d2 : Optional[ComplexFunc], optional, default None
            The value of the second derivative of the background field
            at the point.
        tex : Optional[str], optional, default None
            The LaTeX text of the background field.
        """

        self.__logger: DefaultLogger = DefaultLogger(name)

        self.name: str = name
        self.value: ComplexFunc = value
        self.__value_d: Optional[ComplexFunc] = value_d
        self.__value_d2: Optional[ComplexFunc] = value_d2

        self.tex: str = tex if tex is not None else name

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
            sys.exit(1)

        return self.value(complex(x, 0)).real

    @property
    def value_d(self) -> ComplexFunc:
        """Get the first derivative of the background field at a given
        point.

        Returns
        -------
        ComplexFunc
            The first derivative of the background field.
        """

        if self.__value_d is not None:
            return self.__value_d

        self.__logger.error('Attribute has not been set.')
        sys.exit(1)

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
            sys.exit(1)

        return self.value_d(complex(x, 0)).real

    @property
    def value_d2(self) -> ComplexFunc:
        """Get the second derivative of the background field at a given
        point.

        Returns
        -------
        ComplexFunc
            The second derivative of the background field.
        """

        if self.__value_d2 is not None:
            return self.__value_d2

        self.__logger.error('Attribute has not been set.')
        sys.exit(1)

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
            sys.exit(1)

        return self.value_d2(complex(x, 0)).real
