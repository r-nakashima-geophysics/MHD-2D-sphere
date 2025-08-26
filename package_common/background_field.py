"""A Python module to define a class for handling background fields."""

import sys

from package_common.common_types import ComplexFunc
from package_common.default_logger import DefaultLogger


class BackgroundField:
    """Class to define the profiles of background fields.

    Attributes
    ----------
    name : str
        The name of the background field.
    value : ComplexFunc
         The profile of the background field.
    value_d : ComplexFunc | None
        The first derivative of the profile of the background field.
    value_d2 : ComplexFunc | None
        The second derivative of the profile of the background field.
    tex : str | None
        The LaTeX text of the background field.

    Examples
    --------
    >>> from package_common.background_field import BackgroundField
    >>> linear = BackgroundField('linear', value=lambda x: x)
    >>> linear.r_value(1)
    1.0
    >>> parabola = BackgroundField('parabola', value=lambda x: x**2,
    value_d=lambda x: 2*x, value_d2=lambda x: 2)
    >>> parabola.r_value(1)
    1.0
    >>> parabola.r_value_d(1)
    2.0
    >>> parabola.r_value_d2(1)
    2
    """

    def __init__(self,
                 name: str,
                 *,
                 value: ComplexFunc,
                 value_d: ComplexFunc | None = None,
                 value_d2: ComplexFunc | None = None,
                 tex: str | None = None) -> None:
        """Initialize an instance of the BackgroundField class.

        Parameters
        ----------
        name : str
            The name of the background field.
        value : ComplexFunc
            The profile of the background field.
        value_d : ComplexFunc | None, optional, default None
            The first derivative of the profile of the background field.
        value_d2 : ComplexFunc | None, optional, default None
            The second derivative of the profile of the background field.
        tex : str | None, optional, default None
            The LaTeX text of the background field.
        """

        self.name: str = name
        self.value: ComplexFunc = value
        self.__value_d: ComplexFunc | None = value_d
        self.__value_d2: ComplexFunc | None = value_d2

        self.tex: str
        if tex is not None:
            self.tex = tex
        else:
            self.tex = self.name

        self.__logger: DefaultLogger = DefaultLogger(self.name)

    def r_value(self,
                x: float) -> float:
        """Return the value of the background field at a given (real)
        point.

        Parameters
        ----------
        x : float
            The (real) point at which the value of the background field
            is evaluated.

        Returns
        -------
        float
            The value of the background field at the point.

        Warnings
        --------
        Invalid type of the argument.
            If the argument is not a float or integer.
        """

        if not isinstance(x, (float, int)):
            self.__logger.error('Invalid type of the argument.')
            sys.exit(1)

        return self.value(complex(x, 0)).real

    @property
    def value_d(self) -> ComplexFunc:
        """Return the first derivative of the profile of the background
        field.

        Returns
        -------
        ComplexFunc
            The first derivative of the profile of the background field.

        Warnings
        --------
        This attribute has not been set.
            If the attribute is not set.
        """

        if self.__value_d is not None:
            return self.__value_d

        self.__logger.error('This attribute has not been set.')
        sys.exit(1)

    def r_value_d(self,
                  x: float) -> float:
        """Return the value of the first derivative of the profile of
        the background field at a given (real) point.

        Parameters
        ----------
        x : float
            The (real) point at which the value of the first derivative
            of the profile of the background field is evaluated.

        Returns
        -------
        float
            The value of the first derivative of the profile of the
            background field at the point.

        Warnings
        --------
        Invalid type of the argument.
            If the argument is not a float or integer.
        """

        if not isinstance(x, (float, int)):
            self.__logger.error('Invalid type of the argument.')
            sys.exit(1)

        return self.value_d(complex(x, 0)).real

    @property
    def value_d2(self) -> ComplexFunc:
        """Return the second derivative of the profile of the background
        field.

        Returns
        -------
        ComplexFunc
            The second derivative of the profile of the background
            field.

        Warnings
        --------
        This attribute has not been set.
            If the attribute is not set.
        """

        if self.__value_d2 is not None:
            return self.__value_d2

        self.__logger.error('This attribute has not been set.')
        sys.exit(1)

    def r_value_d2(self,
                   x: float) -> float:
        """Return the value of the second derivative of the profile of
        the background field at a given (real) point.

        Parameters
        ----------
        x : float
            The (real) point at which the value of the second derivative
            of the profile of the background field is evaluated.

        Returns
        -------
        float
            The value of the second derivative of the profile of the
            background field at the point.

        Warnings
        --------
        Invalid type of the argument.
            If the argument is not a float or integer.
        """

        if not isinstance(x, (float, int)):
            self.__logger.error('Invalid type of the argument.')
            sys.exit(1)

        return self.value_d2(complex(x, 0)).real
