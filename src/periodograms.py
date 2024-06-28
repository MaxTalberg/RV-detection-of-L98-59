import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from pickle_data import unpickle_data

# --- Import relevant data
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pickle_file_path = root_dir + "/datasets/cleaned_data_20240531.pickle"
X = unpickle_data(pickle_file_path)
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


# --- GLSP functions
def convert_to_microhertz(frequency_in_days):
    """
    Convert frequency from days to microhertz.

    Parameters:
    -----------
        frequency_in_days (float): Frequency in days.

    Returns:
    --------
        float: Frequency in microhertz.
    """
    return frequency_in_days * 1e6 / 86400


def convert_to_days(frequency_in_microhertz):
    """
    Convert frequency from microhertz to days.

    Parameters:
    -----------
        frequency_in_microhertz (float): Frequency in microhertz.

    Returns:
    --------
        float: Frequency in days.
    """
    return frequency_in_microhertz * 86400 / 1e6


def frequency_to_period(frequency):
    """
    Convert frequency to period.

    Parameters:
    -----------
        frequency (float): Frequency in microhertz.

    Returns:
    --------
        float: Period in days.
    """
    return 1 / (frequency * 86400 * 1e-6)


def compute_periodogram(
    time,
    data,
    min_frequency=min_frequency,
    max_frequency=max_frequency,
    samples_per_peak=samples_per_peak,
    ESPRESSO=False,
):
    """
    Compute the Lomb-Scargle periodogram for given data and time arrays.

    Parameters:
    -----------
        time (array): Time data points.
        data (array): Data points corresponding to time.
        min_frequency (float): Minimum frequency to analyze.
        max_frequency (float): Maximum frequency to analyze.
        samples_per_peak (int): Number of samples per peak for the analysis.
        ESPRESSO (bool): Flag to indicate whether data is from ESPRESSO or not (default False).

    Returns:
    --------
        tuple: Tuple containing frequency array, power array, and FAP thresholds.
    """
    frequency, power = LombScargle(time, data).autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        normalization="standard",
        samples_per_peak=samples_per_peak,
    )
    fap_levels = [0.001, 0.01, 0.1]
    fap_thresholds = LombScargle(time, data).false_alarm_level(fap_levels)
    return frequency, power, fap_thresholds


def plot_periodogram(
    time,
    data,
    activity_indices,
    true_periods,
    period_colors,
    min_frequency=min_frequency,
    max_frequency=max_frequency,
    samples_per_peak=samples_per_peak,
    ESPRESSO=False,
):
    """
    Generate periodogram plots for the given data.

    Parameters:
    -----------
        time (array): Time data points.
        data (DataFrame): Data containing the activity indices.
        activity_indices (list): List of indices to be plotted.
        true_periods (list): True periods for reference lines.
        period_colors (list): Colors for the reference lines.
        min_frequency (float): Minimum frequency to analyze.
        max_frequency (float): Maximum frequency to analyze.
        samples_per_peak (int): Number of samples per peak for the analysis.
        ESPRESSO (bool): Flag to indicate whether data is from ESPRESSO or not (default False).

    Raises:
    -------
        ValueError: If time or data arrays are empty.
        RuntimeError: If computation of periodograms fails.
    """
    if len(time) == 0 or len(data) == 0:
        raise ValueError("Time or data arrays are empty.")

    periodograms = {}

    try:
        for index in activity_indices:
            mask = ~np.isnan(time) & ~np.isnan(data[index])
            cleaned_time = time[mask]
            cleaned_data = data[index][mask]

            if len(cleaned_time) > 0 and len(cleaned_data) > 0:
                frequency, power, fap_thresholds = compute_periodogram(
                    cleaned_time,
                    cleaned_data,
                    min_frequency,
                    max_frequency,
                    samples_per_peak,
                    ESPRESSO,
                )
                periodograms[index] = (frequency, power)
            else:
                raise ValueError(f"Not enough valid data points for index {index}")

        observation_indicator = np.ones_like(time)
        window_frequency, window_power = LombScargle(
            time, observation_indicator
        ).autopower(
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
            normalization="standard",
            samples_per_peak=samples_per_peak,
        )
        periodograms["WF"] = (window_frequency, window_power)

    except Exception as e:
        raise RuntimeError(f"Error computing periodograms: {e}")

    # Plotting
    fig, axes = plt.subplots(len(periodograms), 1, figsize=(10, 20), sharex=True)

    for i, (ax, (key, (frequency, power))) in enumerate(
        zip(axes, periodograms.items())
    ):
        fap_01 = fap_thresholds[0]  # 0.1% FAP level
        fap_1 = fap_thresholds[1]  # 1% FAP level
        fap_10 = fap_thresholds[2]  # 10% FAP level
        frq = convert_to_microhertz(frequency)
        ax.plot(frq, power, color="black", label=key)
        ax.set_ylabel(key, fontsize=20)
        if i < len(axes) - 1:
            for peak, color in zip(true_periods, period_colors):
                ax.axvline(
                    x=convert_to_microhertz(1 / peak),
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                )

        if i < len(axes) - 1:
            ax.axhline(fap_01, color="black", linestyle="dotted", label="0.1% FAP")
            ax.axhline(fap_1, color="black", linestyle="dashdot", label="1% FAP")
            ax.axhline(fap_10, color="black", linestyle="dashed", label="10% FAP")

        if i == 0 and ESPRESSO:
            ax.text(
                -0.1,
                0.46,
                r"$P_{rot}$",
                color="black",
                verticalalignment="top",
                size=18,
            )
            ax.text(
                0.40,
                0.46,
                r"$\frac{P_{rot}}{2}$",
                color="black",
                verticalalignment="top",
                size=18,
            )
            ax.text(1.45, 0.45, r"$P_d$", color="b", verticalalignment="top", size=20)
            ax.text(3.05, 0.45, r"$P_c$", color="g", verticalalignment="top", size=20)
            ax.text(5.04, 0.45, r"$P_b$", color="r", verticalalignment="top", size=20)
            ax.text(6.35, 0.4, "0.1%", color="black", verticalalignment="top", size=20)
            ax.text(6.35, 0.33, "1%", color="black", verticalalignment="top", size=20)
            ax.text(6.35, 0.26, "10%", color="black", verticalalignment="top", size=20)

        if i == 0 and not ESPRESSO:
            ax.text(
                -0.1,
                0.245,
                r"$P_{rot}$",
                color="black",
                verticalalignment="top",
                size=18,
            )
            ax.text(
                0.35,
                0.245,
                r"$\frac{P_{rot}}{2}$",
                color="black",
                verticalalignment="top",
                size=18,
            )
            ax.text(1.45, 0.24, r"$P_d$", color="b", verticalalignment="top", size=20)
            ax.text(3.05, 0.24, r"$P_c$", color="g", verticalalignment="top", size=20)
            ax.text(5.04, 0.24, r"$P_b$", color="r", verticalalignment="top", size=20)
            ax.text(6.35, 0.18, "0.1%", color="black", verticalalignment="top", size=20)
            ax.text(6.35, 0.15, "1%", color="black", verticalalignment="top", size=20)
            ax.text(6.35, 0.12, "10%", color="black", verticalalignment="top", size=20)

    fig.supxlabel("Frequency [µHz]", fontsize=20)
    fig.supylabel("Normalised Power", fontsize=20)

    plt.tight_layout()
    plt.savefig(
        "plots/{}_periodogram.png".format("ESPRESSO" if ESPRESSO else "HARPS")
    )
    print(
        f"Plot saved successfully to output_dir/{'ESPRESSO' if ESPRESSO else 'HARPS'}_periodogram.png"
    )


# --- Plot the periodogram
def run_periodogram_plot(filepath, ESPRESSO=True):
    """Run plotting for given instrument data."""
    # Load data
    X = unpickle_data(filepath)
    X_pre, X_post, X_harps = X["ESPRESSO_pre"], X["ESPRESSO_post"], X["HARPS"]

    if ESPRESSO:
        time = np.concatenate([X_pre["Time"], X_post["Time"]])
        data_df = pd.concat([X_pre, X_post], ignore_index=True)
        activity_indices = ["RV", "FWHM", "BIS", "Sindex", "NaD", "Halpha", "Contrast"]
    else:
        time = X_harps["Time"]
        data_df = X_harps
        activity_indices = ["RV", "Sindex", "NaD", "Halpha", "Hbeta", "Hgamma"]

    # Define periodogram parameters
    true_periods = [2.2531136, 3.6906777, 7.4507245, 78.22, 78.22 / 2]
    period_colors = ["red", "green", "blue", "black", "black"]
    min_frequency = 0.002025227467386652
    max_frequency = 0.5177264355163222
    samples_per_peak = 3000

    # Plot the periodograms
    plot_periodogram(
        time,
        data_df,
        activity_indices,
        true_periods,
        period_colors,
        min_frequency,
        max_frequency,
        samples_per_peak,
        ESPRESSO=ESPRESSO,
    )
