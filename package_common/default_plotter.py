"""A Python module to define a class for figures."""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from package_common.common_types import ArrayAxes, Axes, Figure

if shutil.which('latex') is not None:
    plt.rcParams['text.usetex'] = True
else:
    plt.rcParams['text.usetex'] = False


class DefaultPlotter:
    """Class to handle figures.

    Parameters
    ----------
    nrows : int, optional, default 1
        The number of rows in the figure.
    ncols : int, optional, default 1
        The number of columns in the figure.
    **kwargs
        Keyword variadic arguments.

    Attributes
    ----------
    fig : Figure
        The instance of the Figure class.
    axes : Axes | ArrayAxes
        The instance of the Axes class or the array of them.

    Examples
    --------
    >>> plotter = DefaultPlotter(1, 1)
    >>> x = [1, 2]
    >>> y = [3, 4]
    >>> plotter.axes.plot(x, y)
    >>> plotter.save(Path('.'), 'plot.png', dpi=300)
    """

    def __init__(self,
                 nrows: int = 1,
                 ncols: int = 1,
                 **kwargs) -> None:
        """Initialize an instance of the DefaultPlotter class."""

        self.fig: Figure
        self.axes: Axes | ArrayAxes

        if nrows * ncols == 1:
            axis: Axes
            self.fig, axis = plt.subplots(1, 1, **kwargs)

            axis.grid()
            axis.minorticks_on()

            self.axes = axis
        else:
            axes: ArrayAxes
            self.fig, axes = plt.subplots(nrows, ncols, **kwargs)

            if (nrows == 1) or (ncols == 1):
                num_axes: int
                if ncols == 1:
                    num_axes = nrows
                else:
                    num_axes = ncols

                for i in range(num_axes):
                    axes[i].grid()
                    axes[i].minorticks_on()
            else:
                for i in range(nrows):
                    for j in range(ncols):
                        axes[i, j].grid()
                        axes[i, j].minorticks_on()

            self.axes = axes

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
