"""A Python module to define typed dictionaries for MHD-2D-sphere"""

from pathlib import Path
from typing import TypedDict

from package_common.background_field import BackgroundField


class DictBackgroundField(TypedDict):
    B: BackgroundField
    U: BackgroundField
    NY24: bool


class DictCriterionC(TypedDict):
    degree: int
    ratio: float


class DictFileInfo(TypedDict):
    path_dir: Path
    name_file: str
    name_file_suffix: tuple[str, ...]


class DictParams(TypedDict):
    alpha_init: float
    alpha_end: float
    num_alpha: int
    ohm_max: float
