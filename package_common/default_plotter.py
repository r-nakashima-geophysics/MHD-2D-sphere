"""A Python module to define a class for figures."""

import os
import shutil
from pathlib import Path
from typing import Literal, overload

import matplotlib.pyplot as plt

from package_common.common_types import ArrayAxes, Axes, Figure

if shutil.which('latex') is not None:
    plt.rcParams['text.usetex'] = True
else:
    plt.rcParams['text.usetex'] = False


class DefaultPlotter:
    """Class to handle figures with a single axis.

    Attributes
    ----------
    fig : Figure
        The instance of the Figure class.
    axes : Axes
        The instance of the Axes class.

    Examples
    --------
    >>> x = [1, 2]
    >>> y = [3, 4]
    >>> plotter = DefaultPlotter()
    >>> plotter.axes.plot(x, y)
    >>> plotter.save(Path('.'), 'plot.png', dpi=300)
    """

    def __init__(self,
                 **kwargs) -> None:
        """Initialize an instance of the DefaultPlotter class.

        Parameters
        ----------
        **kwargs
            Keyword variadic arguments.
        """

        self.fig: Figure
        self.axes: Axes
        self.fig, self.axes = plt.subplots(1, 1, **kwargs)

        self.axes.grid()
        self.axes.minorticks_on()

    def save(self,
             path_dir: Path,
             filename: str,
             dpi: int) -> None:
        """Save the figure.

        Parameters
        ----------
        path_dir : Path
            The path of the directory.
        filename : str
            The filename of the figure.
        dpi : int
            The resolution of the figure.
        """

        self.fig.tight_layout()

        os.makedirs(path_dir, exist_ok=True)
        path_fig: Path = path_dir / filename
        self.fig.savefig(path_fig, dpi=dpi)


class DefaultGridPlotter(DefaultPlotter):
    """Subclass of the DefaultPlotter class to handle figures with
    multiple axes.

    Attributes
    ----------
    fig : Figure
        The instance of the Figure class.
    axes : Axes | ArrayAxes
        The instance of the Axes class or the array of the instance of
        the Axes class.

    Examples
    --------
    >>> x = [1, 2]
    >>> y = [3, 4]
    >>> grid_plotter_1_2 = DefaultGridPlotter(1, 2)
    >>> grid_plotter_1_2.axes[0].plot(x, y)
    >>> grid_plotter_2_2 = DefaultGridPlotter(2, 2)
    >>> grid_plotter_2_2.axes[0, 0].plot(x, y)
    """

    def __init__(self,
                 nrows: int = 1,
                 ncols: int = 1,
                 **kwargs) -> None:
        """Initialize an instance of the DefaultGridPlotter class.

        Parameters
        ----------
        nrows : int, optional, default 1
            The number of rows in the figure.
        ncols : int, optional, default 1
            The number of columns in the figure.
        **kwargs
            Keyword variadic arguments.
        """

        if (nrows == 1) and (ncols == 1):
            super().__init__(**kwargs)
        else:
            self.fig: Figure
            self.axes: ArrayAxes
            self.fig, self.axes = plt.subplots(nrows, ncols, **kwargs)

            if (nrows == 1) or (ncols == 1):
                num_axes: int
                if ncols == 1:
                    num_axes = nrows
                else:
                    num_axes = ncols

                for i in range(num_axes):
                    self.axes[i].grid()
                    self.axes[i].minorticks_on()
            else:
                for i in range(nrows):
                    for j in range(ncols):
                        self.axes[i, j].grid()
                        self.axes[i, j].minorticks_on()


@overload
def create_plotter(nrows: Literal[1],
                   ncols: Literal[1],
                   **kwargs) -> DefaultPlotter:
    ...


@overload
def create_plotter(nrows: int,
                   ncols: int,
                   **kwargs) -> DefaultGridPlotter:

    ...


def create_plotter(nrows: int = 1,
                   ncols: int = 1,
                   **kwargs) \
        -> DefaultPlotter | DefaultGridPlotter:
    """Create the instance of the DefaultPlotter class or
    DefaultGridPlotter class.

    Parameters
    ----------
    nrows : int, optional, default 1
        The number of rows in the figure.
    ncols : int, optional, default 1
        The number of columns in the figure.
    **kwargs
        Keyword variadic arguments.

    Returns
    -------
    DefaultPlotter | DefaultGridPlotter
        The instance of the DefaultPlotter class or DefaultGridPlotter
        class.
    """

    if (nrows == 1) and (ncols == 1):
        return DefaultPlotter(**kwargs)

    return DefaultGridPlotter(nrows, ncols, **kwargs)
