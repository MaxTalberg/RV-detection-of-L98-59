
# --- Imports
import warnings
from contextlib import contextmanager
import os
import re
import sys
import time
import copy
import pickle
import radvel
import george
import pypolychord

import numpy as np
import pypolychord.priors as priors

from george import kernels
from pypolychord.settings import PolyChordSettings


import prior_transforms as pt
from utils import load_data_from_pickle

# --- Parameter configuration
INCLUDE_PLANET_B = False

# --- Output directory
output_dir = 'L98_59_aldo/polychord_out'
cluster_dir = 'L98_59_aldo/polychord_out/clusters'

# --- Import relevant data
pickle_file_path = 'datasets/cleaned_data_20240531.pickle' 
X = load_data_from_pickle(pickle_file_path)
X_pre, X_post, X_harps = X['ESPRESSO_pre'], X['ESPRESSO_post'], X['HARPS']
n_pre, n_post, n_harps = len(X_pre['RV']), len(X_post['RV']), len(X_harps['RV'])

# --- Define the parameters
params_general = (['A_RV', 'P_rot', 't_decay', 'gamma',
                   'sigma_RV_pre', 'sigma_RV_post', 'sigma_RV_harps',
                   'v0_pre', 'off_post', 'off_harps'])

params_general_latex = (['A_{RV}', 'P_{rot}', 't_{decay}', '\\gamma',
                    '\\sigma_{RV, pre}', '\\sigma_{RV, post}', '\\sigma_{RV, HARPS}',
                    'v_{0, pre}', 'off_{post}', 'off_{HARPS}'])

params_planet_b = (['P_b', 'secosw_b', 'sesinw_b', 'K_b', 'w*_b', 'phi_b'])
params_planet_c = (['P_c', 'secosw_c', 'sesinw_c', 'K_c', 'w*_c', 'phi_c'])
params_planet_d = (['P_d', 'secosw_d', 'sesinw_d', 'K_d', 'w*_d', 'phi_d'])

params_derived_b = (['e_b', 'Tc_b'])
params_derived_c = (['e_c', 'Tc_c'])
params_derived_d = (['e_d', 'Tc_d'])


if INCLUDE_PLANET_B:
    planet_params = params_planet_b + params_planet_c + params_planet_d
    derived_params = params_derived_b + params_derived_c + params_derived_d

else:
    planet_params = params_planet_c + params_planet_d
    derived_params = params_derived_c + params_derived_d

planet_params_latex = copy.deepcopy(planet_params)

parameters = params_general + planet_params
parameters_latex = params_general_latex + planet_params_latex

Q = {parameters[i]: i for i in range(len(parameters))}

nDims = len(parameters)
nDerived = len(derived_params)

# --- Observational data
# Unpack the data
time_pre, time_post, time_harps = X_pre['Time'], X_post['Time'], X_harps['Time']
obs_pre, obs_post, obs_harps = X_pre['RV'], X_post['RV'], X_harps['RV']
err_pre, err_post, err_harps = X_pre['e_RV'], X_post['e_RV'], X_harps['e_RV']

# Combine all RV data
time_RV = np.block([X_pre['Time'], X_post['Time'], X_harps['Time']])
obs_RV = np.block([X_pre['RV'], X_post['RV'], X_harps['RV']])
err_RV = np.block([X_pre['e_RV'], X_post['e_RV'], X_harps['e_RV']])
adjusted_time_RV = time_RV - 2457000

# --- Scale, offset and jitter
max_jitter_pre = np.median(err_pre)*5
max_jitter_post = np.median(err_post)*5
max_jitter_harps = np.median(err_harps)*5

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

def derived_params(q):

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
    derived_vals = derived_params(q)

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

def mean_fxn(T0, q):
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

def myloglike(theta):

    q = np.copy(theta)

    # Derived parameters
    derived_vals = derived_params(q)

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
    mu_RV = mean_fxn(adjusted_time_RV, q)
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