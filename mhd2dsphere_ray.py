"""A code for 2D ideal MHD waves on a rotating sphere under the
non-Malkus field B_phi = B_0 B(theta) sin(theta)

Plots a figure of the ray trajectory.

Parameters
----------
FILE_PRM : str
    The name of a parameter file

Raises
----------
Too many or too few input parameters in the parameter file
    If the parameter file is inappropriate.
File not found
    If there is not a parameter file with such a name.
Too many input arguments
    If the command line arguments are too many.

Notes
----------
You may save the parameter file in ./input/MHD2Dsphere_ray/.
Parameters other than command line arguments are described below.

References
----------
[1] Ryosuke Nakashima, Shigeo Yoshida, Two-dimensional ideal
magnetohydrodynamic waves on a rotating sphere under a non-Malkus field:
I. Continuous spectrum and its ray-theoretical interpretation.
Geophysical & Astrophysical Fluid Dynamics 118(5-6), 387-440 (2024).
doi: 10.1080/03091929.2024.2384388

Examples
----------
In the below example, the parameters will be set to the default values.
    python3 mhd2dsphere_ray.py
In the below example, the parameter file will be
./input/MHD2Dsphere_ray/prm.dat. (You need not type
'./input/MHD2Dsphere_ray/'.)
    python3 mhd2dsphere_ray prm.dat
"""

import logging
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable, Final

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from package_mhd2dsphere import func_b, func_u

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

FUNC_B: Callable[[complex], complex]
FUNC_DB: Callable[[complex], complex]
TEX_B: str
NAME_B: str

# ========== Parameters ==========

# The boolean value to switch whether to use the magnetostrophic
# approximation
SWITCH_MS: Final[bool] = False

# The function B
# FUNC_B, FUNC_DB, _, TEX_B, NAME_B = func_b.b_hydro('theta')
# FUNC_B, FUNC_DB, _, TEX_B, NAME_B = func_b.b_malkus('theta')
FUNC_B, FUNC_DB, _, TEX_B, NAME_B = func_b.b_sincos('theta')
# FUNC_B, FUNC_DB, _, TEX_B, NAME_B = func_b.b_sin2cos('theta')

# The function U
FUNC_U, FUNC_DU, _, TEX_U, NAME_U = func_u.u_rigid('theta')

# The scaled angular frequency
LAMBDA: Final[float] = 1

# Initial values of the time integration calculating the ray trajectory
THETA_INITIAL: Final[float] = 45
K_INITIAL: Final[float] = 1  # The initial value of scipy.optimize.root
L_INITIAL: Final[float] = 1

# The range of the scaled time
TIME_INIT: Final[float] = 0
TIME_STEP: Final[float] = 0.01
TIME_END: Final[float] = 30

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_ray'
NAME_FIG: Final[str] \
    = f'MHD2Dsphere_ray_{NAME_B}_L{LAMBDA}'
NAME_FIG_SUFFIX: Final[tuple[str, str]] = ('.png', '_ms.png')
FIG_DPI: Final[int] = 600

# ================================

PHI_INIT: Final[float] = 0
THETA_INIT: Final[float] = 0
THETA_END: Final[float] = 180


def load_prm(name_file: str) -> list[float]:
    """Loads a file of parameters

    Parameters
    ----------
    name_file : str
        The name of a parameter file

    Results
    ----------
    prms : list of float
        A list of parameters

    """

    path_file: Path = Path('.') / 'input' \
        / 'MHD2Dsphere_ray' / name_file
    array_prm: np.ndarray = np.loadtxt(path_file, comments='#')

    prms: list[float]
    if array_prm.size == 5:
        prms = array_prm.tolist()
    else:
        logger.error(
            'Too many or too few input parameters '
            + 'in the parameter file')
        sys.exit()

    return prms
#


def wrapper_plot_ray(prms: list[float]) -> None:
    """
    A wrapper of a function to plot a figure of the ray trajectory

    Parameters
    ----------
    prms : list of float
        A list of parameters

    """

    l_init: float
    t_end: float
    _, _, l_init, _, t_end = prms

    results: tuple[float, float, np.ndarray, np.ndarray] \
        = integrate_ray(prms)

    k_const: float
    k_wavenum_init: float
    k_const, k_wavenum_init, _, _ = results

    fig: plt.Figure
    axis: plt.Axes
    axin: list[plt.Axes]
    min_k: float
    max_k: float
    cond_critical: bool
    theta_c_deg: set[float]

    (fig, axis, axin), [min_k, max_k], cond_critical, theta_c_deg \
        = plot_ray(prms, results)

    axis.set_global()

    axis.gridlines(linestyle=':')
    axin[0].grid()
    axin[1].grid()

    axin[0].set_axisbelow(True)
    axin[1].set_axisbelow(True)

    axin[0].set_xlim(min_k, max_k)
    axin[0].set_ylim(min_k, max_k)
    axin[1].set_xlim(TIME_INIT, t_end)
    axin[1].set_ylim(-180, 180)

    axin[0].set_xlabel(r'$k$', fontsize=16)
    axin[0].set_ylabel(r'$l$', fontsize=16)
    axin[1].set_xlabel(
        r'$\mathrm{sgn}(\Omega_0)T=\,$'
        + r'$(|B_0|/R_0\sqrt{\rho_0\mu_\mathrm{m}})t$', fontsize=16)
    axin[1].set_ylabel(
        r'$\mathrm{sgn}(\Omega_0)\phi\,\mathrm{[deg]}$', fontsize=16)

    axin[1].yaxis.set_label_position('right')
    axin[1].yaxis.tick_right()
    axin[1].set_yticks([-180, -90, 0, 90, 180])

    axin[0].tick_params(labelsize=14)
    axin[0].tick_params(labelsize=12, axis="x", labelrotation=45)
    axin[1].tick_params(labelsize=14)
    axin[0].minorticks_on()
    axin[1].minorticks_on()

    axin[0].set_aspect('equal')

    if cond_critical:
        list_lat_c: list[float] \
            = [float(tmp_th) for tmp_th in theta_c_deg]
        sort_lat_c: str = str(sorted(list_lat_c))[1:-1]
        axis.set_title(
            r'$(k_\mathrm{init},l_\mathrm{init})=$'
            + f' ({k_wavenum_init:8.5f}, {l_init}), '
            + r'$\theta_\mathrm{c}=$' + f' {sort_lat_c} [deg]',
            fontsize=16)
    else:
        axis.set_title(
            r'$(k_\mathrm{init},l_\mathrm{init})=$'
            + f' ({k_wavenum_init:8.5f}, {l_init})',
            fontsize=16)

    fig.suptitle(
        r'Ray trajectory [$B_{0\phi}=' + TEX_B + r'$] : '
        + r'$k\sin\theta=$' + f' {k_const:8.5f}, '
        + r'$|\alpha|^{-1/2}\lambda=$' + f' {LAMBDA}',
        fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=1.05)

    os.makedirs(PATH_DIR_FIG, exist_ok=True)

    name_fig_full: str = NAME_FIG
    if FILE_PRM != '':
        name_fig_full += '_' + FILE_PRM.split('.')[0]

    if not SWITCH_MS:
        name_fig_full += NAME_FIG_SUFFIX[0]
    else:
        name_fig_full += NAME_FIG_SUFFIX[1]

    path_fig: Path = PATH_DIR_FIG / name_fig_full

    fig.savefig(path_fig, dpi=FIG_DPI)
#


def plot_ray(prms: list[float],
             results: tuple[float, float, np.ndarray, np.ndarray]) \
        -> tuple[tuple[plt.Figure, plt.Axes, list[plt.Axes]],
                 list[float], bool, set[float]]:
    """Plots a figure of the ray trajectory

    Parameters
    ----------
    prms : list of float
        A list of parameters
    results : tuple of float and ndarray
        A tuple of results

    Returns
    ----------
    fig_bundle : tuple
        A tuple of figures
    [min_k, max_k] : list of float
        The x limits of a graph
    cond_critical : bool
        The boolean value to show whether to exist critical colatitudes
    theta_c_deg : set of float
        The set of critical colatitudes

    """

    theta_deg_init: float = prms[0]

    k_const: float
    lin_time: np.ndarray
    sol_vec_y: np.ndarray
    k_const, _, lin_time, sol_vec_y = results

    num_time: int = len(lin_time)

    fig: plt.Figure
    axis: plt.Axes
    fig, axis = plt.subplots(
        figsize=(7, 6),
        subplot_kw={'projection':
                    ccrs.Mollweide(central_longitude=0.0)})

    axin: list[plt.Axes] = [
        axis.inset_axes((0.12, -0.357, 0.24, 0.32)),
        axis.inset_axes((0.35, -0.37, 0.5, 0.35))
    ]

    phi_deg: np.ndarray = np.rad2deg(sol_vec_y[0])
    theta_rad: np.ndarray = sol_vec_y[1]
    theta_deg: np.ndarray = np.rad2deg(theta_rad)
    l_wavenum: np.ndarray = sol_vec_y[2]

    check_dr: float
    for i_time in range(num_time):
        if i_time % int(num_time/30) == 0:
            check_dr = dispersion(
                k_const, theta_rad[i_time], l_wavenum[i_time])

            print(f' time: {lin_time[i_time]:8.5f}'
                  + f' | phi: {phi_deg[i_time]:>9.4f}'
                  + f' | theta: {theta_deg[i_time]:8.5f}'
                  + f' | l: {l_wavenum[i_time]:8.5f}'
                  + f' | dispersion relation = {check_dr:6.3e}')

    h_kl: np.ndarray = np.zeros_like(phi_deg)
    angle: float
    for i_time in range(num_time):
        angle = math.atan2(
            l_wavenum[i_time], k_const/math.sin(theta_rad[i_time]))
        if angle < 0:
            angle += 2 * math.pi

        h_kl[i_time] = math.degrees(angle)

    lat_deg: np.ndarray = 90 - theta_deg
    lon_deg: np.ndarray = phi_deg - 360*np.floor((phi_deg+180)/360)
    axis.scatter(
        lon_deg, lat_deg, s=0.2, c=h_kl, cmap='hsv',
        vmin=0, vmax=360, transform=ccrs.PlateCarree(), zorder=5)
    axis.scatter(PHI_INIT, 90-theta_deg_init, s=30, c='black',
                 marker='*', transform=ccrs.PlateCarree(), zorder=10)

    max_k: float = 3*math.fabs(k_const)
    min_k: float = -max_k
    grid_k: np.ndarray
    grid_l: np.ndarray
    grid_k, grid_l = np.meshgrid(
        np.linspace(min_k, max_k, 100), np.linspace(min_k, max_k, 100))
    grid_angle: np.ndarray = np.zeros_like(grid_k)
    for i_axin in range(100):
        for j_axin in range(100):
            grid_angle[i_axin, j_axin] = \
                math.atan2(grid_l[i_axin, j_axin],
                           grid_k[i_axin, j_axin])
            if grid_angle[i_axin, j_axin] < 0:
                grid_angle[i_axin, j_axin] += 2 * math.pi

    axin[0].contourf(grid_k, grid_l, grid_angle, cmap='hsv', levels=360)

    axin[1].scatter(lin_time, lon_deg, s=0.2, c=h_kl, cmap='hsv',
                    vmin=0, vmax=360)

    tmp_theta: np.ndarray = np.linspace(THETA_INIT, THETA_END, 18001)
    list_b2: list[float] = [(FUNC_B(tmp)**2).real for tmp in tmp_theta]
    max_b2: float = max(list_b2)
    min_b2: float = min(list_b2)
    cond_critical: bool = \
        (not SWITCH_MS) and (min_b2 < (LAMBDA**2)/(k_const**2) < max_b2)
    theta_c_deg: set[float] = set()
    if cond_critical:
        theta_c_deg = critical_lat(k_const)

    fig_bundle: tuple[plt.Figure, plt.Axes, list[plt.Axes]] \
        = (fig, axis, axin)

    return fig_bundle, [min_k, max_k], cond_critical, theta_c_deg
#


def integrate_ray(prms: list[float]) \
        -> tuple[float, float, np.ndarray, np.ndarray]:
    """Calculates the time integration

    Parameters
    ----------
    prms : list of float
        A list of parameters

    Results
    ----------
    results : tuple of float and ndarray
        A tuple of results

    """

    theta_deg_init: float
    k_init: float
    l_init: float
    t_step: float
    t_end: float
    theta_deg_init, k_init, l_init, t_step, t_end = prms

    theta_rad_init: float = math.radians(theta_deg_init)
    k_const_init: float = k_init * math.sin(theta_rad_init)

    k_const: float = root(
        dispersion, k_const_init,
        args=(theta_rad_init, l_init), method='hybr').x[0]

    k_wavenum_init: float = k_const / math.sin(theta_rad_init)

    t_span: tuple[float, float] = (TIME_INIT, t_end)
    v_init: list[float] = [PHI_INIT, theta_rad_init, l_init]
    num_time: int = 1 + int((t_end-TIME_INIT)/t_step)
    lin_time: np.ndarray = np.linspace(TIME_INIT, t_end, num_time)
    args: list[float] = [k_const]

    sol_vec = solve_ivp(
        main_func, t_span, v_init,
        method='DOP853', t_eval=lin_time, args=args)

    results: tuple[float, float, np.ndarray, np.ndarray] \
        = (k_const, k_wavenum_init, lin_time, sol_vec.y)

    return results
#


def dispersion(k_const: float,
               *args) -> float:
    """The dispersion relation

    Parameters
    ----------
    k_const : float
        The scaled zonal wavenumber
    args : tuple of float
        A tuple of parameters other than k_const (theta_rad, l_wavenum)

    Returns
    ----------
    dispersion_relation : float
        If dispersion_relation = 0, the dispersion relation is satisfied
        for given parameters.

    Notes
    ----------
    This function is based on eq. (30) in Nakashima & Yoshida (submitted)
    [1]_.

    """

    theta_rad: float
    l_wavenum: float
    theta_rad, l_wavenum = args

    value_b: float = FUNC_B(theta_rad).real

    dispersion_relation: float
    if not SWITCH_MS:
        dispersion_relation \
            = ((LAMBDA**2)-(k_const**2)*(value_b**2)) \
            * ((k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2)) \
            + k_const*LAMBDA
    else:
        dispersion_relation \
            = -k_const*(value_b**2) \
            * ((k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2)) \
            + LAMBDA

    return dispersion_relation
#


def main_func(time: np.ndarray,
              vec: np.ndarray,
              k_const: float):
    """A function for the time integration

    Parameters
    ----------
    time : ndarray
        The scaled time
    vec : ndarray
        The integrands (phi, theta, l)
    k_const : float
        The scaled zonal wavenumber

    Returns
    ----------
    d_vec : list of float
        The time integral of the integrands (phi, theta, l)

    """

    _ = time
    theta_rad: float = vec[1]
    l_wavenum: float = vec[2]

    d_vec: list[float] = [d_phi(theta_rad, k_const, l_wavenum),
                          d_theta(theta_rad, k_const, l_wavenum),
                          d_l(theta_rad, k_const, l_wavenum)]

    return d_vec
#


def d_phi(theta_rad: float,
          k_const: float,
          l_wavenum: float) -> float:
    """Calculates the phi component of the group velocity

    Parameters
    ----------
    theta_rad : float
        A colatitude
    k_const : float
        A zonal wavenumber
    l_wavenum : float
        A meridional wavenumber

    Returns
    ----------
    d_phi : float
        the phi component of the group velocity divided by sin(theta)

    Notes
    ----------
    This function is based on eq. (31a) in Nakashima & Yoshida
    (2024)[1]_.

    """

    value_b: float = FUNC_B(theta_rad).real

    cg_phi: float
    if not SWITCH_MS:
        numerator_1: float \
            = 2*k_const*(value_b**2)*math.sin(theta_rad) \
            * (2*(k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2))
        numerator_2: float \
            = (2*k_const*LAMBDA/math.sin(theta_rad)
               + math.sin(theta_rad)) * LAMBDA
        denominator: float \
            = 2*LAMBDA \
            * ((k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2)) \
            + k_const
        cg_phi = (numerator_1-numerator_2) / denominator
    else:
        cg_phi \
            = (value_b**2)*math.sin(theta_rad) \
            * (3*(k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2))

    dphi_dt: float = cg_phi / math.sin(theta_rad)

    return dphi_dt
#


def d_theta(theta_rad: float,
            k_const: float,
            l_wavenum: float) -> float:
    """Calculate the theta component of the group velocity

    Parameters
    ----------
    theta_rad : float
        A colatitude
    k_const : float
        A zonal wavenumber
    l_wavenum : float
        A meridional wavenumber

    Returns
    ----------
    d_theta : float
        the theta component of the group velocity multiplied by -1

    Notes
    ----------
    This function is based on eq. (31b) in Nakashima & Yoshida
    (2024)[1]_.

    """

    value_b: float = FUNC_B(theta_rad).real

    cg_theta: float
    if not SWITCH_MS:
        numerator: float \
            = -2*l_wavenum * ((LAMBDA**2)-(k_const**2)*(value_b**2))
        denominator: float \
            = 2*LAMBDA \
            * ((k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2)) \
            + k_const
        cg_theta = numerator / denominator
    else:
        cg_theta = 2*k_const*l_wavenum*(value_b**2)

    dtheta_dt: float = -cg_theta

    return dtheta_dt
#


def d_l(theta_rad: float,
        k_const: float,
        l_wavenum: float) -> float:
    """Calculates the rate of change of the local meridional wavenumber

    Parameters
    ----------
    theta_rad : float
        A colatitude
    k_const : float
        A zonal wavenumber
    l_wavenum : float
        A meridional wavenumber

    Returns
    ----------
    d_l : float
        the rate of change of the local meridional wavenumber

    Notes
    ----------
    This function is based on eq. (32c) in Nakashima & Yoshida
    (2024)[1]_.

    """

    value_b: float = FUNC_B(theta_rad).real
    value_db: float = FUNC_DB(theta_rad).real

    dlambda_dtheta: float
    if not SWITCH_MS:
        numerator: float = -k_const*LAMBDA/math.tan(theta_rad) + (
            2*(k_const**2)*value_b*value_db
            + 2*(k_const**2)*(value_b**2)/math.tan(theta_rad)
        ) * ((k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2))
        denominator: float \
            = 2*LAMBDA \
            * ((k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2)) \
            + k_const
        dlambda_dtheta = numerator / denominator
    else:
        dlambda_dtheta = k_const*(
            2*value_b*value_db + (value_b**2)/math.tan(theta_rad)
        ) * ((k_const**2)/(math.sin(theta_rad)**2)+(l_wavenum**2))

    dl_dt: float = -d_phi(theta_rad, k_const, l_wavenum) \
        * k_const/math.tan(theta_rad) + dlambda_dtheta

    return dl_dt
#


def critical_lat(k_const: float) -> set[float]:
    """Calculates critical latitudes

    Parameters
    ----------
    k_const : float
        A scaled zonal wavenumber

    Returns
    ----------
    set_theta_c : set of float
        critical latitudes

    """

    def critical(theta_rad: float) -> float:
        value_b: float = FUNC_B(theta_rad).real
        critical: float = (LAMBDA**2) - (k_const**2)*(value_b**2)

        return critical

    set_theta_c: set = set()

    init_rad: float
    theta_c_rad: float
    theta_c_deg: float
    for init_deg in range(int(THETA_INIT), int(THETA_END), 1):

        init_rad = math.radians(init_deg)
        theta_c_rad = root(critical, init_rad, method='hybr').x[0]
        theta_c_deg = math.degrees(theta_c_rad)

        if THETA_INIT < theta_c_deg < THETA_END:
            set_theta_c.add(f'{theta_c_deg:4.2f}')

    return set_theta_c
#


if __name__ == '__main__':
    TIME_START: Final[float] = perf_counter()

    FILE_PRM: str
    list_prm: list[float]
    if len(sys.argv) == 1:
        FILE_PRM = ''
        list_prm = [
            THETA_INITIAL, K_INITIAL, L_INITIAL, TIME_STEP, TIME_END
        ]
    elif len(sys.argv) == 2:
        FILE_PRM = sys.argv[1]
        path_file_prm = Path('.') / 'input' \
            / 'MHD2Dsphere_ray' / FILE_PRM
        if not os.path.exists(path_file_prm):
            logger.error('File not found')
            sys.exit()

        list_prm = load_prm(FILE_PRM)
    else:
        logger.error('Too many input arguments')
        sys.exit()

    plt.rcParams['text.usetex'] = True

    wrapper_plot_ray(list_prm)

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_START
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
