import radvel
import numpy as np

from config_params import Q, INCLUDE_PLANET_B

adjusted_time_RV = np.array([
    1408.8536609997973,
    1930.5773693597876
])

def compute_derived_params(q):

    # Time 
    T_start = min(adjusted_time_RV)
    T_end = max(adjusted_time_RV)


    # --- Derived parameters

    # planet b
    if INCLUDE_PLANET_B:
        e_b = np.sqrt(q[Q['secosw_b']]**2 + q[Q['sesinw_b']]**2)
        Tc_b = q[Q['phi_b']] * (T_end - T_start) + T_start

    # planet c
    e_c = np.sqrt(q[Q['secosw_c']]**2 + q[Q['sesinw_c']]**2)
    Tc_c = q[Q['phi_c']] * (T_end - T_start) + T_start

    # planet d
    e_d = np.sqrt(q[Q['secosw_d']]**2 + q[Q['sesinw_d']]**2)
    Tc_d = q[Q['phi_d']] * (T_end - T_start) + T_start



    if INCLUDE_PLANET_B:
        return [e_b, Tc_b, e_c, Tc_c, e_d, Tc_d]
    else:
        return [e_c, Tc_c, e_d, Tc_d]


def compute_planets_RV(T0, q):

    RV_total = np.zeros(len(T0))
    derived_vals = compute_derived_params(q)

    if INCLUDE_PLANET_B:
        e_b, Tc_b, e_c, Tc_c, e_d, Tc_d = derived_vals
    
        # Initialize RadVel parameters with correct mapping
        radvel_params = radvel.Parameters(3, basis='per tc e w k', planet_letters={1:'b', 2:'c', 3:'d'})

        # planet b
        radvel_params['per1'] = radvel.Parameter(value=q[Q['P_b']])
        radvel_params['tc1'] = radvel.Parameter(value=Tc_b)
        radvel_params['e1'] = radvel.Parameter(value=e_b)
        radvel_params['w1'] = radvel.Parameter(value=q[Q['w*_b']])
        radvel_params['k1'] = radvel.Parameter(value=q[Q['K_b']])

        # planet c
        radvel_params['per2'] = radvel.Parameter(value=q[Q['P_c']])
        radvel_params['tc2'] = radvel.Parameter(value=Tc_c)
        radvel_params['e2'] = radvel.Parameter(value=e_c)
        radvel_params['w2'] = radvel.Parameter(value=q[Q['w*_c']])
        radvel_params['k2'] = radvel.Parameter(value=q[Q['K_c']])

        # planet d
        radvel_params['per3'] = radvel.Parameter(value=q[Q['P_d']])
        radvel_params['tc3'] = radvel.Parameter(value=Tc_d)
        radvel_params['e3'] = radvel.Parameter(value=e_d)
        radvel_params['w3'] = radvel.Parameter(value=q[Q['w*_d']])
        radvel_params['k3'] = radvel.Parameter(value=q[Q['K_d']])

    else:
        e_c, Tc_c, e_d, Tc_d = derived_vals

        # Initialize RadVel parameters with correct mapping
        radvel_params = radvel.Parameters(2, basis='per tc e w k', planet_letters={1:'c', 2:'d'})

        # planet c
        radvel_params['per1'] = radvel.Parameter(value=q[Q['P_c']])
        radvel_params['tc1'] = radvel.Parameter(value=Tc_c)
        radvel_params['e1'] = radvel.Parameter(value=e_c)
        radvel_params['w1'] = radvel.Parameter(value=q[Q['w*_c']])
        radvel_params['k1'] = radvel.Parameter(value=q[Q['K_c']])

        # planet d
        radvel_params['per2'] = radvel.Parameter(value=q[Q['P_d']])
        radvel_params['tc2'] = radvel.Parameter(value=Tc_d)
        radvel_params['e2'] = radvel.Parameter(value=e_d)
        radvel_params['w2'] = radvel.Parameter(value=q[Q['w*_d']])
        radvel_params['k2'] = radvel.Parameter(value=q[Q['K_d']])

    # Make sure to use a model setup correctly
    model = radvel.RVModel(radvel_params)

    RV_total += model(T0)
    
    return RV_total

def compute_offset(T0, q):
    n_pre = 39
    n_post = 24
    Y0 = np.zeros(T0.shape)

    # ESPRESSO pre offset
    Y0[0:n_pre] += q[Q['v0_pre']]
    # ESPRESSO post offset
    Y0[n_pre:n_pre + n_post] += q[Q['v0_pre']] + q[Q['off_post']]
    # HARPS offset
    Y0[n_pre + n_post:] += q[Q['v0_pre']] + q[Q['off_harps']]
    # Compute the RV model with corrected offsets
    Y0 += compute_planets_RV(T0, q)
    return Y0
