import numpy as np
import prior_transforms as pt
from config_params import Q, INCLUDE_PLANET_B

# Prior transformation function: Transforms uniform hypercube to the parameter space
def planet_prior(qq):

    if INCLUDE_PLANET_B:
        # planet b
        qq[Q['P_b']] = pt.jeffreys(qq[Q['P_b']], 0.1, 520)
        qq[Q['secosw_b']] = pt.kipping_beta(qq[Q['secosw_b']])
        qq[Q['sesinw_b']] = pt.kipping_beta(qq[Q['sesinw_b']])
        qq[Q['K_b']] = pt.uniform(qq[Q['K_b']], 0, 17)
        qq[Q['w*_b']] = pt.uniform(qq[Q['w*_b']], -np.pi, np.pi)
        qq[Q['phi_b']] = pt.uniform(qq[Q['phi_b']], 0, 1)

    # planet c
    qq[Q['P_c']] = pt.jeffreys(qq[Q['P_c']], 0.1, 520)
    qq[Q['secosw_c']] = pt.kipping_beta(qq[Q['secosw_c']])
    qq[Q['sesinw_c']] = pt.kipping_beta(qq[Q['sesinw_c']])
    qq[Q['K_c']] = pt.uniform(qq[Q['K_c']], 0, 17)
    qq[Q['w*_c']] = pt.uniform(qq[Q['w*_c']], -np.pi, np.pi)
    qq[Q['phi_c']] = pt.uniform(qq[Q['phi_c']], 0, 1)

    # planet d
    qq[Q['P_d']] = pt.jeffreys(qq[Q['P_d']], 0.1, 520)
    qq[Q['secosw_d']] = pt.kipping_beta(qq[Q['secosw_d']])
    qq[Q['sesinw_d']] = pt.kipping_beta(qq[Q['sesinw_d']])
    qq[Q['K_d']] = pt.uniform(qq[Q['K_d']], 0, 17)
    qq[Q['w*_d']] = pt.uniform(qq[Q['w*_d']], -np.pi, np.pi)
    qq[Q['phi_d']] = pt.uniform(qq[Q['phi_d']], 0, 1)

    return qq
