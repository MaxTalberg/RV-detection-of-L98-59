#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  ---  Imports
 
import os.path
import sys
import pickle
import time
import socket
import numpy as np
import re
import copy
from shutil import copyfile

import pypolychord
from pypolychord.settings import PolyChordSettings
import pypolychord.priors as py_pt

sys.path.insert(1,'../../software')

import GP_soft as GP
import prior_transforms as pt
import RV_soft as RV
import util_soft as util
scale_and_offset = util.scale_and_offset
nparr = np.array

# ---  Top-level parameter control

# Format e.g.: 101_bcde_a (index, planets, sub-index)
RUN_IDX = sys.argv[1] 
RUN_SAMPLER = True
USE_GP = False # <---------- CHECKME
INCLUDE_HIRES = False # <---------- CHECKME

NLE = -1e30 # null log evidence
M_STAR = 0.905 

# --- Output stuff

hpc_dir = '/rds/user/vr325/hpc-work'

run_identifier = ''.join(['K10_', RUN_IDX])
output_dir = ''.join([hpc_dir, '/K10_aldo/polychord_out/', run_identifier,''])
cluster_dir = output_dir + '/clusters'
summary_dir = ''.join([hpc_dir, '/K10_aldo/polychord_summaries'])

for my_dir in [output_dir, cluster_dir, summary_dir]:
   if not os.path.exists(my_dir): os.makedirs(output_dir, exist_ok=True) 

# --- Kepler-10 - parameters for known (transiting) planets from D14

P_b, sig_P_b = 0.8374907, 2e-7
Tc_b, sig_Tc_b = 55034.08687, 1.8e-4

P_c, sig_P_c = 45.294301, 4.8e-5
Tc_c, sig_Tc_c = 55062.26648, 8.1E-04

# --- Define model parameters to be fit
if USE_GP:
    params_general = (['sRV_add1', 'sRV_add2', 
        'sHK_add', 'sBS_add',
        'V_r', 'V_c', 'L_c', 'B_c', 'B_r',
        'P', 'lambda_p', 'lambda_e',
        'p0_RV1', 'p0_RV2',
        'p0_HK', 'p0_BS'])
    params_general_latex = ([
        r'\sigma^{+}_{RV1}', r'\sigma^{+}_{RV2}', 
        r'\sigma^{+}_{HK}', r'\sigma^{+}_{BS}',
        'V_r', 'V_c', 'L_c', 'B_c', 'B_r',
        'P', r'\lambda_p', r'\lambda_e',
        r'p0_{RV1}', r'p0_{RV2}',
         r'p0_{HK}', r'p0_{BS}'])
else:
    params_general = (['sRV_add1', 'sRV_add2', 'p0_RV1', 'p0_RV2'])
    params_general_latex = ([r'\sigma^{+}_{RV1}', r'\sigma^{+}_{RV2}', 
        r'p0_{RV1}', r'p0_{RV2}',
        ])

if INCLUDE_HIRES:
    params_general.extend(['sRV_addH', 'p0_RVH'])
    params_general_latex.extend([r'\sigma^{+}_{RVH}', r'p0_{RV,HRS}'])

    # if USE_GP:
    #     params_general.extend(['sHK_add_HRS', 'p0_HK_HRS'])
    #     params_general_latex.extend([r'\sigma^{+}_{HK,HRS}', r'p0_{HK,HRS}'])

# --- Parameter management

run_idx_split = RUN_IDX.split('_')
n_planets = len(run_idx_split[1])

# Parameters for any fixed planets, e.g. from transits
params_trans_planets = []

# Populate parameters of fixed planets
planets_list, n_trans_planets = run_idx_split[1], 0

if 'b' in planets_list:
    params_trans_planets.extend(['P_b', 'Tc_b', 'K_b', 'e_b', 'w_b'])
    n_trans_planets += 1
if 'c' in planets_list:
    params_trans_planets.extend(['P_c', 'Tc_c', 'K_c', 'e_c', 'w_c'])
    n_trans_planets += 1

params_trans_planets_latex = copy.deepcopy(params_trans_planets)
n_free_planets = n_planets-n_trans_planets

# P = period; K = semi-ampl.; e = ecc; w = omega; M = mean anomaly
params_free_planets = ['P', 'K', 'e', 'w', 'M']*(n_free_planets)
params_free_planets_latex = copy.deepcopy(params_free_planets)

# Add index to label parameters for different planets
for i, param in enumerate(params_free_planets):
    params_free_planets[i] += '{:.0f}'.format(np.ceil((i+1)/5))
    params_free_planets_latex[i] += '_{:.0f}'.format(np.ceil((i+1)/5))

# Store list of all parameters; compute problem dimensionality
parameters = params_general + params_trans_planets + params_free_planets
parameters_latex = (params_general_latex + params_trans_planets_latex 
    + params_free_planets_latex)
n_params = len(parameters)

# Make dictionary that translates between parameter name and index in q vector
# (letter q used as stand-in for Greek letter theta)
Q = {parameters[i]: i for i in range(len(parameters))}

# -- Set PolyChord sampling parameters
algorithm_params = {'do_clustering': True,
                    'precision_criterion': 1e-9,
                    'num_repeats': 5*n_params,
                    'read_resume': True,
                    'nprior': 5000,
                    'nfail': 50000,
                    }
output_params =     {'base_dir': output_dir,
                    'file_root': run_identifier,
                    'feedback': 3}


settings = PolyChordSettings(n_params, 0, 
    **{**algorithm_params, **output_params})

# ---  Import and pre-process observed data



# -- Import relevant data

X = util.load_dict('../data/data_preproc_20211110.pickle')
X1, X2, XH = X['X1'], X['X2'], X['H1'] # HARPSN-1, HARPSN-2, HIRES

n1, n2, nh = len(X1['vrad']), len(X2['vrad']), len(XH['vrad'])
# sub-arrays: obs_RV[0:n1], obs_RV[n1:n1+n2], obs_RV[n1+n2::]

# Re-package data into consistently-named arrays

if INCLUDE_HIRES:
    jdb_RV = np.block([X1['jdb'], X2['jdb'], XH['jdb']])
    obs_RV = np.block([X1['vrad'], X2['vrad'], XH['vrad']])
    err_RV = np.block([X1['svrad'], X2['svrad'], XH['svrad']])
else:
    jdb_RV = np.block([X1['jdb'], X2['jdb']])
    obs_RV = np.block([X1['vrad'], X2['vrad']])
    err_RV = np.block([X1['svrad'], X2['svrad']])


jdb_HK = np.block([X1['jdb'], X2['jdb']])
obs_HK = np.block([X1['rhk'], X2['rhk']])
err_HK = np.block([X1['sig_rhk'], X2['sig_rhk']])

jdb_BS = np.block([X1['jdb'], X2['jdb']])
obs_BS = np.block([X1['rhk'], X2['rhk']])
err_BS = np.block([X1['sig_rhk'], X2['sig_rhk']])

# Get scales for each time series
scale_RV1, scale_sRV1, offset_RV1 = scale_and_offset(obs_RV, err_RV)
scale_HK, scale_sHK, offset_HK = scale_and_offset(obs_HK, err_HK)
scale_BS, scale_sBS, offset_BS = scale_and_offset(obs_BS, err_BS)

# -- Perform initial offset fitting to activity time series

# p, V = np.polyfit(jdb_HK,obs_HK,0,cov=True)
# p0_HK, sp0_HK = p, np.sqrt(V)[0][0]

# p, V = np.polyfit(jdb_BS,obs_BS,0,cov=True)
# p0_BS, sp0_BS = p, np.sqrt(V)[0][0]

jitter_RV_max = np.sqrt(scale_RV**2 - np.sum(err_RV**2)/len(err_RV))
jitter_HK_max = np.sqrt(scale_HK**2 - np.sum(err_HK**2)/len(err_HK))
jitter_BS_max = np.sqrt(scale_BS**2 - np.sum(err_BS**2)/len(err_BS))


scale_RV_HRS, scale_sRV_HRS, offset_RV_HRS = scale_and_offset(
    obs_RV_HRS, err_RV_HRS, N_scale=1)

scale_HK_HRS, scale_sHK_HRS, offset_HK_HRS = scale_and_offset(
    obs_HK_HRS, err_HK_HRS, N_scale=1)

jitter_RV_HRS_max = np.sqrt(scale_RV_HRS**2 - 
    np.sum(err_RV_HRS**2)/len(err_RV_HRS))
jitter_HK_HRS_max = np.sqrt(scale_HK_HRS**2 - 
    np.sum(err_HK_HRS**2)/len(err_HK_HRS))

N_DRS, N_HRS = len(obs_RV), len(obs_RV_HRS)

# Tile to form full set of observations
jdb_tuple = ([jdb_RV, jdb_HK, jdb_BS]) if USE_GP else jdb_RV
obs_big = np.block([obs_RV, obs_HK, obs_BS]) if USE_GP else obs_RV


# --- Define parameter prior functions (unit hyper-q --> param. values)

def planet_prior(qq):

    if 'b' in planets_list:
        qq[Q['P_b']] = pt.gaussian(qq[Q['P_b']], P_b, sig_P_b)
        qq[Q['Tc_b']] = pt.gaussian(qq[Q['Tc_b']], Tc_b, sig_Tc_b)
        qq[Q['K_b']] = pt.modjeffreys(qq[Q['K_b']], 1, 5)
        qq[Q['e_b']] = pt.half_gaussian(qq[Q['e_b']], 0.098)
        qq[Q['w_b']] = pt.uniform(qq[Q['w_b']], 0, 2*np.pi)

    if 'c' in planets_list:
        qq[Q['P_c']] = pt.gaussian(qq[Q['P_c']], P_c, sig_P_c)
        qq[Q['Tc_c']] = pt.gaussian(qq[Q['Tc_c']], Tc_c, sig_Tc_c)
        qq[Q['K_c']] = pt.modjeffreys(qq[Q['K_c']], 1, 5)
        qq[Q['e_c']] = pt.half_gaussian(qq[Q['e_c']], 0.098)
        qq[Q['w_c']] = pt.uniform(qq[Q['w_c']], 0, 2*np.pi)

    if n_free_planets > 0:
        qq[Q['P1']::5] = py_pt.LogSortedUniformPrior(0.5,3000)(qq[Q['P1']::5])

        for n in range(n_free_planets):
            qq[Q['K1']+5*n] = pt.modjeffreys(qq[Q['K1']+5*n], 1, 5)
            qq[Q['e1']+5*n] = pt.half_gaussian(qq[Q['e1']+5*n], 0.098)
            qq[Q['w1']+5*n] = pt.uniform(qq[Q['w1']+5*n], 0, 2*np.pi)
            qq[Q['M1']+5*n] = pt.uniform(qq[Q['M1']+5*n], 0, 2*np.pi)
    
    return qq

def myprior(q):

    qq = np.copy(q) ## Polychord-NB - make copy of input first; NB qq vs q

    # Additive white noise for RV
    qq[Q['sRV_add1']] = pt.modjeffreys(q[Q['sRV_add2']], .001*scale_sRV,
     jitter_RV_max)

    qq[Q['sRV_add1']] = pt.modjeffreys(q[Q['sRV_add2']], .001*scale_sRV,
     jitter_RV_max)
    
    if USE_GP:
        # Additive white noise for RHK, BIS
        qq[Q['sHK_add']] = pt.modjeffreys(q[Q['sHK_add']], .001*scale_sHK,
             jitter_HK_max)                                   
        qq[Q['sBS_add']] = pt.modjeffreys(q[Q['sBS_add']], .001*scale_sBS,
             jitter_BS_max) 
                
        qq[Q['V_r']] = pt.half_gaussian(q[Q['V_r']], scale_RV/2) 
        qq[Q['V_c']] = pt.gaussian(q[Q['V_c']], 0, scale_RV/2) 
        qq[Q['L_c']] = pt.gaussian(q[Q['L_c']], 0, scale_HK)
        qq[Q['B_c']] = pt.gaussian(q[Q['B_c']], 0, scale_BS)
        qq[Q['B_r']] = pt.gaussian(q[Q['B_r']], 0, scale_BS)

        # Quasi-periodic GP hyper-parameters
        qq[Q['P']] = pt.uniform(q[Q['P']], 15, 75)
        qq[Q['lambda_p']] = pt.jeffreys(q[Q['lambda_p']], 0.1, 10)
        qq[Q['lambda_e']] = pt.jeffreys(q[Q['lambda_e']], 20, 200) 

    # Mean function DC pffset for RV
    qq[Q['p0_RV1']] = pt.gaussian(q[Q['p0_RV1']], 0, 2)
    qq[Q['p0_RV2']] = pt.gaussian(q[Q['p0_RV2']], 0, 2)
    
    if USE_GP:
        # Mean function DC offsets for RHK, BIS
        qq[Q['p0_HK']] = pt.gaussian(q[Q['p0_HK']], p0_HK, sp0_HK)  
        qq[Q['p0_BS']] = pt.gaussian(q[Q['p0_BS']], p0_BS, sp0_BS)

    if INCLUDE_HIRES:
        qq[Q['sRV_addH']] = pt.modjeffreys(q[Q['sRV_addH']], 
            .001*scale_sRV, jitter_RV_max)
        qq[Q['p0_RVH']] = pt.gaussian(q[Q['p0_RVH']], 0, 2)

        # if USE_GP:
        #     qq[Q['sHK_add_HRS']] = pt.modjeffreys(q[Q['sHK_add_HRS']], 
        #         .001*scale_sHK_HRS, jitter_HK_HRS_max)
        #     qq[Q['p0_HK_HRS']] = pt.gaussian(q[Q['p0_HK_HRS']], p0_HK_HRS,
        #         sp0_HK_HRS)

    # Planet parameters
    if n_planets > 0:
        qq = planet_prior(qq)

    return qq

# --- Define mean functions for various time series, incl. Keplerians (RVs)

def compute_planets_RV(T0, q):

    RV_total = np.zeros(len(T0))

    # Below: calculate periapse passage time since this will depend on
    # both e and omega (can fix T0 if e=0, in which case omega irrelevant)
    if 'b' in planets_list:

        # Kepler-10 b RV signal; omeg_b set to zero (arbitrary but fixed)
        T0_B = RV.transit_to_periapse(q[Q['P_b']], q[Q['Tc_b']], 0, 0)
        q_b = nparr([q[Q['P_b']], q[Q['K_b']], 0, 0, T0_B ])
        RV_total += RV.kepler_RV(T0, q_b, periapse_time_provided=True)

    if 'c' in planets_list:
        T0_c = RV.transit_to_periapse(q[Q['P_c']], q[Q['Tc_c']], q[Q['w_c']], 
            q[Q['e_c']])
        q_c = nparr([q[Q['P_c']], q[Q['K_c']], q[Q['e_c']], q[Q['w_c']], T0_c])
        RV_total += RV.kepler_RV(T0, q_c, periapse_time_provided=True)

    # RV signal due to any other planets in model
    RV_total += RV.kepler_RV(T0, q[Q['P1']::])

    return  RV_total

def mean_fxn(T_in, q):
    '''
    Compute mean functions given three types of observations: RV, LHK, BS, 
    observed at times T0, T1, T2; assuming constant DC offset for each, and
    0, 1, 2, ... Keplerian terms for RV only. For RV and LHK series, separate
    mean terms for HIRES and HARPS-N observations.
    '''

    if USE_GP:
        (T0, T1, T2) = T_in
        Y0, Y1, Y2 = np.zeros(T0.shape), np.zeros(T1.shape), np.zeros(T2.shape)
    else:
        T0 = T_in
        Y0 = np.zeros(T0.shape)        

    # DC offset for RVs
    Y0[0:n1] += q[Q['p0_RV']]
    Y0[n1:n1+n2] += q[Q['p0_RV1']]

    if include_HIRES: Y0[n1+n2::] += q[Q['p0_RVH']]

    if n_planets > 0: Y0 += compute_planets_RV(T0, q)

    if USE_GP:

        # DC offset for log R'_HK
        Y1[0:N_DRS] += q[Q['p0_HK']]
        # if INCLUDE_HIRES: Y1[N_DRS::] += q[Q['p0_HK_HRS']]

        # DC offset for BIS
        Y2 += q[Q['p0_BS']] # DC offset for BIS

        return [Y0, Y1, Y2]

    else:
        return Y0

# --- Define parameter likelihood function

def check_stability(P_all, K_all, e_all):

    for i in range(n_planets):
        for j in range(i+1, n_planets):
            a_i = RV.get_semi_axes(P_all[i], M_STAR, e_all[i], 1, 1)
            a_j = RV.get_semi_axes(P_all[j], M_STAR, e_all[j], 1, 1)
            m_i = RV.get_msini(P_all[i], K_all[i], e_all[i], M_STAR, 1, 1)
            m_j = RV.get_msini(P_all[i], K_all[i], e_all[i], M_STAR, 1, 1)
            r_hill = RV.get_hill_radius(m_i, m_j, a_i[0], a_j[0], M_STAR, 1)

            if RV.get_gladman_stability(a_i[0], a_j[0], r_hill) == 0: 
                return False

            # Check larger major axes correspond to larger minor axes
            minor_flag = np.sign(a_i[0]-a_j[0])*np.sign(a_i[1]-a_j[1])
            if minor_flag < 0: 
                return False

    return True # if no stability conditions violated above

def myloglike(theta):
    """ Compute Gaussian log likelihood; first computes GP covariance
    matrix and Kepler model prediction. NB if using Polychord, must always
    return (logL, [derived_params])
    """

    q = np.copy(theta) ## Polychord-NB - make copy of input first
    
    # Only allow finite parameters (Gaussian prior can cause problems!)
    if not np.all(np.isfinite(q)): return NLE, []

    if n_planets > 0:
 
        planet_periods_trans = []

        if 'b' in planets_list: planet_periods_trans.append(P_b)
        if 'c' in planets_list: planet_periods_trans.append(P_c)

        planet_periods_free = ([q[value] for key, value in Q.items() 
            if re.match(r'^P\d$|^P_[a-z]$', key)])

        planets_P = planet_periods_trans + planet_periods_free

        planets_K = ([q[value] for key, value in Q.items() 
            if re.match(r'^K\d$|^K_[a-z]$', key)])
        
        planets_e = ([q[value] for key, value in Q.items() 
            if re.match(r'^e\d$|^e_[a-z]$', key)])

        # Impose  eccentricity constraint: e < 1
        if any(nparr(planets_e) >= 1): 
            return NLE, [] # Null_log_evidence

        if n_planets > 1:
            # Check dynamical stability (Gladman condition)
            if not check_stability(planets_P, planets_K, planets_e): 
                return NLE, []

    # Compute observational error including additive white noise jitter

# X1['svrad'], X2['svrad'], XH['svrad']

    if INCLUDE_HIRES:
        err_RV_add = np.block([ 
            np.sqrt(X1['svrad']**2 + q[Q['sRV_add1']]**2),
            np.sqrt(X2['svrad']**2 + q[Q['sRV_add2']]**2), 
            np.sqrt(XH['svrad']**2 + q[Q['sRV_addH']]**2) 
            ])
    else:
        err_RV_add = np.block([ 
            np.sqrt(X1['svrad']**2 + q[Q['sRV_add1']]**2),
            np.sqrt(X2['svrad']**2 + q[Q['sRV_add2']]**2), 
            ])
        
    # if (USE_GP and INCLUDE_HIRES):
    #     err_HK_add = np.block([ np.sqrt(err_HK**2 + q[Q['sHK_add']]**2), 
    #         np.sqrt(err_HK_HRS**2 + q[Q['sHK_add_HRS']]**2)])
    if USE_GP:
        err_HK_add = np.sqrt(err_HK**2 + q[Q['sHK_add']]**2)

    sig_obs_tuple = [err_RV_add, err_HK_add, err_BS] if USE_GP else err_RV_add


    if USE_GP:
        # Ensure GP evolution time is longer than single period [Rajpaul thesis]
        LE_min = np.sqrt(3*(q[Q['P']]**2)*(q[Q['lambda_p']]**2)/2/np.pi)
        if  q[Q['lambda_e']] < LE_min:  return NLE, []

        # Compute GP covariance excluding additive W.N. 
        # (already added for RV, HK above) -- CHECKME
        QP_params = np.concatenate((nparr([0,0]),q[3:12]))

        # Compute GP covariance matrix including additive white noise
        K_trial = GP.K_QP_activity(jdb_tuple, QP_params, 
            sig_obs= sig_obs_tuple)
        # Compute mean function
        mu_RV, mu_HK, mu_BS = mean_fxn(jdb_tuple, q)
        y_res = obs_big-np.block([mu_RV,mu_HK,mu_BS])
        logL = GP.logL_GP(y_res, K_trial)
    else:    
        K_trial = GP.K_WN(jdb_tuple, 0, 
                          sig_obs_tuple)
        # Compute mean function
        mu_RV = mean_fxn(jdb_tuple, q)
        y_res = obs_big-mu_RV
        logL = GP.logL_GP(y_res, K_trial, K_is_diag=True)

    return logL, [] # Polychord-NB: must return second []

# --- Run MCMC sampler

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

if RUN_SAMPLER:
    # %% ---  Run MultiNest sampler, aggregate outputs, and store to disk
    
    # Run PolyChord
    t_start = time.time()

    output = pypolychord.run_polychord(myloglike, n_params, 0, settings, 
        myprior, dumper)
    output.make_paramnames_files(list(zip(parameters,parameters_latex)))

    t_end = time.time()
    
    # Compute total runtime for Polychord run
    runtime = t_end-t_start
    
    # Make a getdist plot
    try:
        import getdist.plots
        posterior = output.posterior
        g = getdist.plots.getSubplotPlotter()
        g.triangle_plot(posterior, filled=True)
        g.export(''.join([summary_dir,'/', run_identifier,'.pdf']))
    except ImportError:
        print("Install matplotlib and getdist for plotting examples")
    #except:
    #    print('Other exception occured while trying to make triangle plot')

    # Pickle files of interest
    mcmc_output = ({'Q': Q,
        'output': output, 'runtime': runtime, 'hostname': socket.gethostname(),
         'output_params': output_params, 'algorithm_params': algorithm_params})

    pickle_file_name = ''.join([summary_dir, '/',
        run_identifier,'.pickle'])
    if runtime > 25: # avoid saving runtime for short re-started runs
        with open(pickle_file_name, 'wb') as handle:
           pickle.dump(mcmc_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Copy posterior files & summary stats to summary directory
    extensions = ['.txt', '.stats', '_equal_weights.txt']
    for ext in extensions:
        copyfile(''.join([output_dir,'/',output_params['file_root'],ext]),
            ''.join([summary_dir,'/', output_params['file_root'],ext]))

    print('Finished running PolyChord')