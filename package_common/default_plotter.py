"""A Python module to define a class for figures.

Notes
-----
The Initializers of the DefaultPlotter class and the DefaultGridPlotter
class should not be used to create instances of these classes directly.
The use of `create_plotter` function is recommended.
"""

import os
import shutil
from pathlib import Path
from typing import Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import axes, collections, colorbar, figure, legend

from package_common.default_logger import DefaultLogger

type Figure = figure.Figure
type Axes = axes.Axes
type Legend = legend.Legend
type PathCollection = collections.PathCollection
type Colorbar = colorbar.Colorbar

type ArrayAxes = npt.NDArray[np.object_]
type ArrayLegend = npt.NDArray[np.object_]
type ArrayPathCollection = npt.NDArray[np.object_]


class DefaultPlotter:
    """Class to handle figures with a single axis.

    Attributes
    ----------
    fig : Figure
        The instance of the Figure class.
    axes : Axes
        The instance of the Axes class.
    leg : Legend | None
        The instance of the Legend class.
    sc : PathCollection | None
        The instance of the PathCollection class.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from package_common.default_plotter import DefaultPlotter
    >>> x = [1, 2]
    >>> y = [3, 4]
    >>> plotter = DefaultPlotter()
    >>> plotter.axes.plot(x, y)
    >>> plotter.tight_layout()
    >>> plotter.save(Path('.'), 'plot.png', dpi=300)
    >>> plt.show()
    """

    __set_latex: bool = False

    def __init__(self,
                 **kwargs) -> None:
        """Initialize an instance of the DefaultPlotter class.

        Parameters
        ----------
        **kwargs
            Keyword variadic arguments.
        """

        DefaultPlotter.set_latex()

        self.fig: Figure
        self.axes: Axes
        self.fig, self.axes = plt.subplots(1, 1, **kwargs)

        self.leg: Legend | None = None
        self.sc: PathCollection | None = None

        self.axes.grid()
        self.axes.set_axisbelow(True)
        self.axes.minorticks_on()

    def save(self,
             path_dir: Path,
             filename: str,
             dpi: int = 300,
             /,
             switch_tight_layout: bool = True) -> None:
        """Save the figure.

        Parameters
        ----------
        path_dir : Path
            The path of the directory.
        filename : str
            The filename of the figure.
        dpi : int, optional, default 300
            The resolution of the figure.
        switch_tight_layout : bool, optional, default True
            The boolean value to switch whether to use tight layout or
            not.
        """

        if not isinstance(self.leg, np.ndarray):
            if self.leg is not None:
                self.leg.get_frame().set_alpha(1)
        else:
            for leg in self.leg.ravel():
                if leg is not None:
                    leg.get_frame().set_alpha(1)

        if switch_tight_layout:
            self.fig.tight_layout()

        os.makedirs(path_dir, exist_ok=True)
        path_fig: Path = path_dir / filename
        self.fig.savefig(path_fig, dpi=dpi)

        DefaultLogger(filename).info('Saved')

    def tight_layout(self) -> None:
        """Adjust the padding of the figure."""

        self.fig.tight_layout()

    @classmethod
    def set_latex(cls) -> None:
        """Set latex."""

        if not cls.__set_latex:
            cls.__set_latex = True

            if shutil.which('latex') is not None:
                plt.rcParams['text.usetex'] = True
            else:
                plt.rcParams['text.usetex'] = False


class DefaultGridPlotter(DefaultPlotter):
    """Subclass of the DefaultPlotter class to handle figures with
    multiple axes.

    Attributes
    ----------
    fig : Figure
        The instance of the Figure class.
    axes : ArrayAxes
        The array of the instance of the Axes class.
    leg : ArrayLegend
        The array of the instance of the Legend class.
    sc : ArrayPathCollection
        The array of the instance of the PathCollection class.

    Examples
    --------
    Create a (1, 2) plot:
        >>> import matplotlib.pyplot as plt
        >>> from package_common.default_plotter import DefaultGridPlotter
        >>> x = [1, 2]
        >>> y = [3, 4]
        >>> grid_plotter = DefaultGridPlotter(1, 2)
        >>> grid_plotter.axes[0].plot(x, y)
        >>> grid_plotter.tight_layout()
        >>> grid_plotter.save(Path('.'), 'plot.png', dpi=300)
        >>> plt.show()
    Create a (2, 2) plot:
        >>> grid_plotter = DefaultGridPlotter(2, 2)
        >>> grid_plotter.axes[0, 0].plot(x, y)
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
            return

        DefaultPlotter.set_latex()

        self.fig: Figure
        self.axes: ArrayAxes
        self.fig, self.axes = plt.subplots(nrows, ncols, **kwargs)

        self.leg: ArrayLegend \
            = np.full_like(self.axes, None, dtype=np.object_)
        self.sc: ArrayPathCollection \
            = np.full_like(self.axes, None, dtype=np.object_)

        for axis in self.axes.ravel():
            axis.grid()
            axis.set_axisbelow(True)
            axis.minorticks_on()


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
                   **kwargs) -> DefaultPlotter | DefaultGridPlotter:
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

    Examples
    --------
    >>> from package_common.default_plotter import create_plotter
    >>> plotter = create_plotter()
    >>> grid_plotter = create_plotter(2, 2)
    """

    if (nrows == 1) and (ncols == 1):
        return DefaultPlotter(**kwargs)

    return DefaultGridPlotter(nrows, ncols, **kwargs)
