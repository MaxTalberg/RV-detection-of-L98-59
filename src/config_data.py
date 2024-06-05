import numpy as np
from pickle_data import load_data_from_pickle

# --- Import relevant data
pickle_file_path = 'datasets/cleaned_data_20240531.pickle' 
X = load_data_from_pickle(pickle_file_path)
X_pre, X_post, X_harps = X['ESPRESSO_pre'], X['ESPRESSO_post'], X['HARPS']
n_pre, n_post, n_harps = len(X_pre['RV']), len(X_post['RV']), len(X_harps['RV'])

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
