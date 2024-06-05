import copy
from config_loader import load_config

config = load_config("config.ini")

# --- Parameter configuration
INCLUDE_PLANET_B = config["Parameters"].getboolean("INCLUDE_PLANET_B")

params_general = [
    "A_RV",
    "P_rot",
    "t_decay",
    "gamma",
    "sigma_RV_pre",
    "sigma_RV_post",
    "sigma_RV_harps",
    "v0_pre",
    "off_post",
    "off_harps",
]

params_general_latex = [
    "A_{RV}",
    "P_{rot}",
    "t_{decay}",
    "\\gamma",
    "\\sigma_{RV, pre}",
    "\\sigma_{RV, post}",
    "\\sigma_{RV, HARPS}",
    "v_{0, pre}",
    "off_{post}",
    "off_{HARPS}",
]

params_planet_b = ["P_b", "secosw_b", "sesinw_b", "K_b", "w*_b", "phi_b"]
params_planet_c = ["P_c", "secosw_c", "sesinw_c", "K_c", "w*_c", "phi_c"]
params_planet_d = ["P_d", "secosw_d", "sesinw_d", "K_d", "w*_d", "phi_d"]

params_derived_b = ["e_b", "Tc_b"]
params_derived_c = ["e_c", "Tc_c"]
params_derived_d = ["e_d", "Tc_d"]

planet_params = params_planet_c + params_planet_d
derived_params = params_derived_c + params_derived_d

if INCLUDE_PLANET_B:
    planet_params = params_planet_b + planet_params
    derived_params = params_derived_b + derived_params


planet_params_latex = copy.deepcopy(planet_params)

parameters = params_general + planet_params
parameters_latex = params_general_latex + planet_params_latex

Q = {parameters[i]: i for i in range(len(parameters))}

nDims = len(parameters)
nDerived = len(derived_params)
