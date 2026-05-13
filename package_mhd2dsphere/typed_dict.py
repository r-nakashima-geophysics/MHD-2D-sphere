"""A Python module to define typed dictionaries for MHD-2D-sphere"""

from pathlib import Path
from typing import TypedDict

from package_common.background_field import BackgroundField
from package_common.common_types import ArrayComplex, ArrayFloat, ArrayStr
from package_common.spectral_deform import ComplexCoordinate


class DictBackgroundField(TypedDict):
    """Typed dictionary for background fields."""

    B: BackgroundField
    U: BackgroundField
    MU: ComplexCoordinate
    NY24: bool


class DictCriterionC(TypedDict):
    """Typed dictionary for convergence criteria."""

    degree: int
    ratio: float


class DictResult(TypedDict):
    """Typed dictionary for the results of the eigenvalue problem."""

    lin_alpha: ArrayFloat | None
    eig_val: ArrayComplex | None
    eig_vec: ArrayComplex | None
    pke: ArrayFloat | None
    pme: ArrayFloat | None
    psm: ArrayFloat | None
    pse: ArrayFloat | None
    ohm: ArrayFloat | None
    sym: ArrayStr | None


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
