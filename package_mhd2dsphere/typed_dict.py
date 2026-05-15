"""A Python module to define typed dictionaries for MHD-2D-sphere"""

from pathlib import Path
from typing import TypedDict

from package_common.background_field import BackgroundField
from package_common.common_types import ArrayComplex, ArrayFloat, ArrayStr
from package_common.spectral_deform import ComplexCoordinate
from package_common.utils_collocation import ChebyshevGaussQuad


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
    eig: ArrayComplex
    vec_psi: ArrayComplex | None
    vec_vpa: ArrayComplex | None
    phys_qtys: DictPhysQtys


class DictPhysQtys(TypedDict):
    """Typed dictionary for physical quantities."""

    pke: ArrayFloat
    pme: ArrayFloat
    psm: ArrayFloat
    pse: ArrayFloat
    ohm: ArrayFloat
    sym: ArrayStr


class DictChebyshevGaussQuad(TypedDict):
    """Typed dictionary for Chebyshev-Gauss quadrature."""

    quad_pke: ChebyshevGaussQuad
    quad_pme: ChebyshevGaussQuad
    quad_psm: ChebyshevGaussQuad
    quad_pse_u: ChebyshevGaussQuad
    quad_pse_b: ChebyshevGaussQuad
    quad_ohm: ChebyshevGaussQuad


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
