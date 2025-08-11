"""A Python module to define a class for figures."""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from package_common.common_types import Any, ArrayAxes, Axes, Figure

if shutil.which('latex') is not None:
    plt.rcParams['text.usetex'] = True
else:
    plt.rcParams['text.usetex'] = False


class DefaultPlotter:
    """Class to handle figures.

    Attributes
    ----------
    fig : Figure
        The instance of the figure.
    axes : Axes
        The instance of the axes.
    """

    def __init__(self,
                 nrows: int = 1,
                 ncols: int = 1,
                 **kwargs) -> None:
        """Initialize the DefaultPlotter instance.

        Parameters
        ----------
        nrows : int, optional, default 1
            The number of rows in the figure.
        ncols : int, optional, default 1
            The number of columns in the figure.
        **kwargs : dict[str, Any], optional
            Keyword variadic arguments.
        """

        self.fig: Figure
        self.axes: Axes | ArrayAxes
        self.fig, self.axes = plt.subplots(nrows, ncols, **kwargs)

        if nrows * ncols == 1:
            self.axes.grid()
            self.axes.minorticks_on()
        elif (nrows == 1) or (ncols == 1):
            num_axes: int = nrows if ncols == 1 else ncols
            for i in range(num_axes):
                self.axes[i].grid()
                self.axes[i].minorticks_on()
        else:
            for i in range(nrows):
                for j in range(ncols):
                    self.axes[i, j].grid()
                    self.axes[i, j].minorticks_on()

    def save(self,
             path_dir: Path,
             path_fig: str,
             dpi: int) -> None:
        """Save the figure.

        Parameters
        ----------
        path_dir : Path
            The path of the directory.
        path_fig : str
            The filename of the figure.
        dpi : int
            The resolution of the figure.
        """

        self.fig.tight_layout()

        os.makedirs(path_dir, exist_ok=True)
        path: Path = path_dir / path_fig
        self.fig.savefig(path, dpi=dpi)
