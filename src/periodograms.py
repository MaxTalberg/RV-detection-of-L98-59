import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from pickle_data import load_data_from_pickle

# --- Import relevant data
pickle_file_path = "datasets/cleaned_data_20240531.pickle"
X = load_data_from_pickle(pickle_file_path)
X_pre, X_post, X_harps = X["ESPRESSO_pre"], X["ESPRESSO_post"], X["HARPS"]
n_pre, n_post, n_harps = len(X_pre["RV"]), len(X_post["RV"]), len(X_harps["RV"])

# --- Concatenate data
ESPRESSO_df = pd.concat([X_pre, X_post], ignore_index=True)
ESPRESSO_times = np.block([X_pre["Time"], X_post["Time"]])
HARPS_df = X_harps
HARPS_times = X_harps["Time"]
time = np.concatenate([ESPRESSO_times, HARPS_times])

# --- GLSP parameters
samples_per_peak = 3000

min_frequency = 0.002025227467386652
max_frequency = 0.5177264355163222

ESPRESSO_activity_indices = ["RV", "FWHM", "BIS", "Sindex", "NaD", "Halpha", "Contrast"]
HARPS_activity_indices = [
    "RV",
    "FWHM",
    "BIS",
    "Sindex",
    "NaD",
    "Halpha",
    "Hbeta",
    "Hgamma",
]

true_periods = [2.2531136, 3.6906777, 7.4507245]  # D. S. Demangeon et al. 2021
period_colors = ["red", "green", "blue"]

# --- GLSP functions


# convert frequency to microhertz
def convert_to_microhertz(frequency_in_days):
    return frequency_in_days * 1e6 / 86400


# Perform the Lomb-Scargle periodogram
def compute_periodogram(
    time,
    data,
    min_frequency=min_frequency,
    max_frequency=max_frequency,
    samples_per_peak=samples_per_peak,
):

    frequency, power = LombScargle(time, data).autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        normalization="standard",
        samples_per_peak=samples_per_peak,
    )
    return frequency, power


def plot_GLSP(times, df, min_freq, max_freq, HARPS=False):

    if HARPS:
        activity_indices = HARPS_activity_indices
        inst = "HARPS"
    else:
        activity_indices = ESPRESSO_activity_indices
        inst = "ESPRESSO"
    # Compute periodograms
    periodograms = {}
    for index in activity_indices:
        frequency, power = compute_periodogram(times, df[index], min_freq, max_freq)
        periodograms[index] = (frequency, power)

    # Compute the window function
    observation_indicator = np.ones_like(times)
    window_power = LombScargle(times, observation_indicator).power(frequency)
    periodograms["WF"] = (frequency, window_power)

    # Plot the periodograms
    fig, axes = plt.subplots(len(periodograms), 1, figsize=(10, 10), sharex=True)

    for i, (ax, (key, (frequency, power))) in enumerate(
        zip(axes, periodograms.items())
    ):
        ax.plot(convert_to_microhertz(frequency), power, color="black", label=key)
        ax.set_ylabel(key)
        if i < len(axes) - 1:
            for peak, color in zip(true_periods, period_colors):
                ax.axvline(
                    x=convert_to_microhertz(1 / peak),
                    color=color,
                    linestyle="--",
                    label=f"True Period: {peak} days",
                )

        if i == 0 and HARPS:
            ax.text(1.2, 0.18, r"$P_d$", color="b", verticalalignment="top", size=12)
            ax.text(2.8, 0.18, r"$P_c$", color="g", verticalalignment="top", size=12)
            ax.text(4.80, 0.18, r"$P_b$", color="r", verticalalignment="top", size=12)
        elif i == 0:
            ax.text(1.2, 0.27, r"$P_d$", color="b", verticalalignment="top", size=12)
            ax.text(2.8, 0.27, r"$P_c$", color="g", verticalalignment="top", size=12)
            ax.text(4.80, 0.27, r"$P_b$", color="r", verticalalignment="top", size=12)
    fig.supxlabel("Frequency [µHz]")
    fig.supylabel("Normalised Power")

    plt.tight_layout()
    plt.savefig(f"plots/GLSP_{inst}.png")


# --- Plot the GLSP for the ESPRESSO and HARPS data
plot_GLSP(ESPRESSO_times, ESPRESSO_df, min_frequency, max_frequency)
plot_GLSP(HARPS_times, HARPS_df, min_frequency, max_frequency, HARPS=True)
