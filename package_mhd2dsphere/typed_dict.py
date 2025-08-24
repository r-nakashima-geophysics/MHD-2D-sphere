"""A Python module to define typed dictionaries for MHD-2D-sphere"""

from pathlib import Path
from typing import TypedDict

from package_common.background_field import BackgroundField


class DictBackgroundField(TypedDict):
    """Typed dictionary for background fields."""

    B: BackgroundField
    U: BackgroundField
    MU: BackgroundField
    NY24: bool


class DictCriterionC(TypedDict):
    """Typed dictionary for convergence criteria."""

    degree: int
    ratio: float


class DictFileInfo(TypedDict):
    """Typed dictionary for file information."""

    path_dir: Path
    name_file: str
    name_file_suffix: tuple[str, ...]


class DictParams(TypedDict):
    """Typed dictionary for parameters."""

    alpha_init: float
    alpha_end: float
    num_alpha: int
    ohm_max: float
