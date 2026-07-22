# MHD-2D-sphere

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10730383.svg)](https://doi.org/10.5281/zenodo.10730383)

These Python scripts support the findings of our study, Nakashima &amp; Yoshida (2024).

> Ryosuke Nakashima, Shigeo Yoshida, Two-dimensional ideal
> magnetohydrodynamic waves on a rotating sphere under a non-Malkus field:
> I. Continuous spectrum and its ray-theoretical interpretation.
> Geophysical & Astrophysical Fluid Dynamics 118(5-6), 387-440 (2024).
> doi: [10.1080/03091929.2024.2384388](https://doi.org/10.1080/03091929.2024.2384388)

| Figure                 |  mhd2dsphere\_\*.py |
| :--------------------- | ------------------: |
| 2, 3                   |              malkus |
| 4, 6                   |         eig, eigfig |
| 5                      |       sincos_degree |
| 7, 8, A4               |         sincos_eigf |
| 9, 10, 11              |      sincos_alleigf |
| 12                     |            harmonic |
| 13, 14, 15, 16         |               local |
| 17, 18, 19, 20, 21, A3 |                 ray |
| A1, A8                 |          sincos_fmr |
| A2                     |      sincos_fmreigf |
| A5                     |    sincos_frobenius |
| A6                     | sincos_allfrobenius |

## Setup

It is recommended to run these scripts with Python 3.14 or later. First, install the required Python packages using the following command:

Additionally, the scripts require the common package, `r-nakashima-geophysics/common-package` ([https://github.com/r-nakashima-geophysics/common-package.git](https://github.com/r-nakashima-geophysics/common-package.git)). For example, place the common package in the same parent directory as the script:

```sh
git clone https://github.com/r-nakashima-geophysics/common-package.git
cp -r common-package/package_common .
```

The recommended version of the common package is v1.0.13 or later.

## Usage

```sh
python3 mhd2dsphere_*.py
```

See the docstring of each Python script for details.
