# --- Imports
import george
import pypolychord

import numpy as np
from george import kernels
from pypolychord.settings import PolyChordSettings

import prior_transforms as pt

from config_params import nDims, nDerived, Q
from config_data import err_harps, err_post, err_pre, adjusted_time_RV, obs_RV, max_jitter_post, max_jitter_pre

from priors import planet_prior
from computations import compute_derived_parameters, compute_offset


# --- Output directory
output_dir = 'L98_59_aldo/polychord_out'
cluster_dir = 'L98_59_aldo/polychord_out/clusters'


def myprior(q):

    qq = np.copy(q)

    # priors 
    qq[Q['A_RV']] = pt.uniform(q[Q['A_RV']], 0, 17) # U(0, 17)
    qq[Q['P_rot']] = pt.jeffreys(q[Q['P_rot']], 5, 520) # J(5, 520)
    qq[Q['t_decay']] = pt.jeffreys(q[Q['t_decay']], qq[Q['P_rot']] / 2, 2600)     # T_decay > P_rot/2 + J(2.5, 2600)
    qq[Q['gamma']] = pt.uniform(q[Q['gamma']], 0.05, 5) # U(0.05, 5)
    qq[Q['sigma_RV_pre']] = pt.uniform(q[Q['sigma_RV_pre']], 0, max_jitter_pre) # U(0, max_jitter_pre)
    qq[Q['sigma_RV_post']] = pt.uniform(q[Q['sigma_RV_post']], 0, max_jitter_post)  # U(0, max_jitter_post)
    qq[Q['v0_pre']] = pt.gaussian(q[Q['v0_pre']], -5579.1, 0.0035)  # N(-5579.1, 0.0035)
    qq[Q['off_post']] = pt.gaussian(q[Q['off_post']], 2.88, 4.8)    # N(2.88, 4.8)
    qq[Q['off_harps']] = pt.gaussian(q[Q['off_harps']], -99.5, 5.0)  # N(-99.5, 5.0)

    # planet priors
    qq = planet_prior(qq)

    return qq


def myloglike(theta):

    q = np.copy(theta)

    # Derived parameters
    derived_vals = compute_derived_parameters(q)

    # Compute the GP covariance matrix
    log_period = np.log(1.0 / q[Q['P_rot']])    # frequency = 1 / period

    #print("Log period (frequency = 1 / P_rot):", log_period)
    #log_period = np.log(q[Q['P_rot']])
    K_trial = q[Q['A_RV']]**2 * kernels.ExpSine2Kernel(gamma=q[Q['gamma']], log_period=log_period) * kernels.ExpSquaredKernel(metric=q[Q['t_decay']]**2)
    gp = george.GP(K_trial)

    # Compute the RV error
    err_RV = np.block([
        np.sqrt(err_pre**2 + q[Q['sigma_RV_pre']]**2),
        np.sqrt(err_post**2 + q[Q['sigma_RV_post']]**2),
        np.sqrt(err_harps**2 + q[Q['sigma_RV_harps']]**2)
    ])

    # Compute the GP model
    gp.compute(adjusted_time_RV, err_RV)

    # Compute the RV model and residuals
    mu_RV = compute_offset(adjusted_time_RV, q)
    residuals = obs_RV - mu_RV

    # Compute the log likelihood
    log_likelihood = gp.log_likelihood(residuals)

    #chi_squared = np.sum((residuals / err_RV)**2)
    #log_likelihood = -0.5 * chi_squared

    return log_likelihood, derived_vals


def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])
    

# --- Set up PolyChord settings
algorithm_params = {'do_clustering': True,
                    'precision_criterion': 10,
                    'num_repeats': 1,
                    'read_resume': False,
                    'nprior': 500,
                    'nfail': 5000,
                    }

output_params =     {'base_dir': output_dir,
                     'feedback': 0}

settings = PolyChordSettings(nDims, nDerived, **algorithm_params)

output = pypolychord.run_polychord(
    loglikelihood=myloglike,
    nDims=nDims,
    nDerived=nDerived,
    prior=myprior,
    dumper=dumper,
    settings=settings
)
