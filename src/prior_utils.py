import numpy as np
from pickle_data import unpickle_data


def derive_priors(pickle_file_path):

    # Load data from pickle file
    X = unpickle_data(pickle_file_path)
    X_pre, X_post, X_harps = X["ESPRESSO_pre"], X["ESPRESSO_post"], X["HARPS"]

    # Extract time, observations, and errors
    time_pre, time_post, time_harps = X_pre["Time"], X_post["Time"], X_harps["Time"]
    obs_pre, obs_post, obs_harps = X_pre["RV"], X_post["RV"], X_harps["RV"]
    err_pre, err_post, err_harps = X_pre["e_RV"], X_post["e_RV"], X_harps["e_RV"]

    # Combine all RV data
    time_RV = np.block([time_pre, time_post, time_harps])
    obs_RV = np.block([obs_pre, obs_post, obs_harps])
    err_RV = np.block([err_pre, err_post, err_harps])
    adjusted_time_RV = time_RV - 2457000

    # Print RV related information
    print("############## RV ##############")
    print("vo offset:", np.median(obs_pre), "std", np.sum(err_pre))
    print(
        "pre,post offset:",
        np.median(obs_post) - np.median(obs_pre),
        "std",
        np.sqrt(np.std(obs_post) ** 2 + np.std(obs_pre) ** 2),
    )
    print(
        "pre,harps offset:",
        np.median(obs_harps) - np.median(obs_pre),
        "std",
        np.sqrt(np.std(obs_harps) ** 2 + np.std(obs_pre) ** 2),
    )

    print("A_RV max:", np.max([np.ptp(obs_pre), np.ptp(obs_post), np.ptp(obs_harps)]))
    print("A_RV all max:", np.max(np.ptp(obs_RV)))

    # FWHM and S-index processing
    fwhm_obs_pre, fwhm_obs_post = X_pre["FWHM"] / 1000, X_post["FWHM"] / 1000
    fwhm_err_pre, fwhm_err_post = X_pre["e_FWHM"], X_post["e_FWHM"]

    fwhm_obs = np.block([fwhm_obs_pre, fwhm_obs_post])
    print("############## FWHM ##############")
    print("C fwhm pre:", np.median(fwhm_obs_pre), "std", np.std(fwhm_obs_pre))
    print("C fwhm post:", np.median(fwhm_obs_post), "std", np.std(fwhm_obs_post))
    print("A fwhm max:", np.max([np.ptp(fwhm_obs_pre), np.ptp(fwhm_obs_post)]))

    sindex_obs_pre, sindex_obs_post, sindex_obs_harps = (
        X_pre["Sindex"],
        X_post["Sindex"],
        X_harps["Sindex"],
    )
    sindex_all = np.block([sindex_obs_pre, sindex_obs_post, sindex_obs_harps])
    print("############## S-index ##############")
    print("C sindex pre:", np.median(sindex_obs_pre), "std", np.std(sindex_obs_pre))
    print("C sindex post:", np.median(sindex_obs_post), "std", np.std(sindex_obs_post))
    print(
        "C sindex harps:", np.median(sindex_obs_harps), "std", np.std(sindex_obs_harps)
    )
    print("A max Sindex:", np.max(np.ptp(sindex_all)))

    # Return a dictionary of combined and processed data for further use
    return {
        "time_RV": time_RV,
        "obs_RV": obs_RV,
        "err_RV": err_RV,
        "adjusted_time_RV": adjusted_time_RV,
        "fwhm_obs": fwhm_obs,
        "sindex_all": sindex_all,
    }


# Usage
pickle_file_path = "datasets/cleaned_data_20240531.pickle"
data = derive_priors(pickle_file_path)
