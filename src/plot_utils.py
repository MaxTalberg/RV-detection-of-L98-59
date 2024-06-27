import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from george import kernels, GP
from astropy.stats import LombScargle

from pickle_data import unpickle_data
from periodograms import convert_to_microhertz


class PlotUtils:
    """
    A utility class for processing radial velocity (RV) data for astronomical observations.

    **Attributes**

    ``pickle_file_path`` : str
        The file path to the pickle file containing observational data.
    ``adjusted_time_RV`` : numpy.ndarray
        Adjusted observational times accounting for the base time subtraction.
    ``obs_RV_adjusted`` : numpy.ndarray
        Observed radial velocities adjusted for specific instrument offsets.
    ``err_RV`` : numpy.ndarray
        Errors associated with the observed radial velocities.

    **Methods**

    __init__(self, pickle_file_path)
        Initializes the object, loads data, and sets up Gaussian Processes.
    load_data(self)
        Loads and processes observational data from a pickle file.
    setup_gp(self)
        Sets up the Gaussian Processes for subsequent analysis.
    """

    def __init__(self, pickle_file_path):
        """
        Initializes the PlotUtils object.

        Parameters
        ----------
        pickle_file_path : str
            Path to the pickle file containing the observational data.
        """
        self.pickle_file_path = pickle_file_path
        self.load_data()
        self.setup_gp()

    def load_data(self):
        """
        Load and process data from a pickle file for astronomical observations.

        This method initializes attributes for observational times, radial velocities (RV),
        and associated errors, adjusting for specific instrument offsets and baseline times.

        The adjustments are made to:
        - Align time measurements to a specific baseline (BJD 2457000).
        - Correct observed RVs for known instrumental offsets.
        - Combine data from multiple instruments.

        Raises
        ------
        FileNotFoundError
            If the pickle file cannot be found.
        ValueError
            If data extraction from the pickle file fails.
        """
        try:
            X = unpickle_data(self.pickle_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.pickle_file_path} was not found.")
        except Exception as e:
            raise ValueError(f"Failed to load or parse the pickle file: {str(e)}")

        try:
            # Extracting observational data
            self.time_pre, self.time_post, self.time_harps = (
                X["ESPRESSO_pre"]["Time"],
                X["ESPRESSO_post"]["Time"],
                X["HARPS"]["Time"],
            )
            self.obs_pre, self.obs_post, self.obs_harps = (
                X["ESPRESSO_pre"]["RV"],
                X["ESPRESSO_post"]["RV"],
                X["HARPS"]["RV"],
            )
            err_pre, err_post, err_harps = (
                X["ESPRESSO_pre"]["e_RV"],
                X["ESPRESSO_post"]["e_RV"],
                X["HARPS"]["e_RV"],
            )

            # Adjusting times and RVs for baseline and known offsets
            self.adjusted_time_RV = (
                np.block([self.time_pre, self.time_post, self.time_harps]) - 2457000
            )
            self.obs_RV_adjusted = np.block(
                [
                    self.obs_pre + 5578.506526225824,
                    self.obs_post + 5578.506526225824 - 2.000428193875587,
                    self.obs_harps + 5578.506526225824 + 99.50156136523518,
                ]
            )
            self.err_RV = np.block(
                [
                    np.sqrt(err_pre**2 + 1.2702928849665707**2),
                    np.sqrt(err_post**2 + 0.5959691795036054**2),
                    np.sqrt(err_harps**2 + 0.27246571959879184**2),
                ]
            )
        except KeyError as e:
            raise KeyError(f"Key error in data dictionary: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing data: {str(e)}")

    def setup_gp(self):
        """
        Configures and computes a Gaussian Process (GP) model with specified kernel parameters for radial velocity data.

        The GP uses a combination of an exponential sine squared kernel for periodic components and an
        exponential squared kernel for decay processes, suitable for modeling stellar activity and instrumental noise.

        Updates several class attributes used for storing the Gaussian Process model and results from its predictions.

        Raises
        ------
        Exception
            If there is an error in setting up or computing the Gaussian Process.

        Attributes
        ----------
        gp : george.GP
            The Gaussian Process model fitted with observational data.
        time_dense : numpy.ndarray
            Dense array of time points for making smooth GP predictions.
        mu : numpy.ndarray
            Predicted mean of the GP model at each point in `time_dense`.
        var : numpy.ndarray
            Predicted variance of the GP model at each point in `time_dense`.
        std : numpy.ndarray
            Standard deviation derived from `var`.
        mu_t : numpy.ndarray
            Predicted mean of the GP model at the actual observational times.
        var_t : numpy.ndarray
            Predicted variance of the GP model at the actual observational times.
        std_t : numpy.ndarray
            Standard deviation derived from `var_t`.
        """
        try:
            # Define the GP kernel parameters
            log_period = np.log(
                27.38068137156724
            )  # Natural log of the period, useful for periodic kernels
            A_RV = 7.711836893488454  # Amplitude of the RV variations
            gamma = 2.917125126168333  # Factor in the exponential sine squared kernel
            t_decay = (
                382.801299740658  # Timescale of decay in the exponential squared kernel
            )

            # Setup the Gaussian Process kernel
            kernel = (
                A_RV
                * kernels.ExpSine2Kernel(gamma=gamma, log_period=log_period)
                * kernels.ExpSquaredKernel(metric=t_decay)
            )

            # Initialize the Gaussian Process
            self.gp = GP(kernel)
            self.gp.compute(
                self.adjusted_time_RV, self.err_RV
            )  # Computing the GP with observational data and errors

            # Time points for detailed GP predictions
            self.time_dense = np.linspace(
                min(self.adjusted_time_RV), max(self.adjusted_time_RV), 1000
            )
            self.mu, self.var = self.gp.predict(
                self.obs_RV_adjusted, self.time_dense, return_var=True
            )
            self.std = np.sqrt(self.var)

            # Predictions at actual observational times for residuals analysis
            self.mu_t, self.var_t = self.gp.predict(
                self.obs_RV_adjusted, self.adjusted_time_RV, return_var=True
            )
            self.std_t = np.sqrt(self.var_t)

        except Exception as e:
            # Capture any type of exception from the Gaussian Process setup and raise as a general error
            raise Exception(f"Failed to setup or compute Gaussian Process: {str(e)}")

    def plot_gp(self):
        """
        Plots the Gaussian Process model results and the residuals.

        The first subplot displays the Gaussian Process model fit along with the observational data.
        The second subplot shows the residuals from the GP fit.
        This method visualizes the GP fitting and the discrepancies between the model and observed data.
        """
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(15, 10),
            sharey="row",
            gridspec_kw={"height_ratios": [7, 3], "width_ratios": [8, 2]},
        )
        axs[0, 0].errorbar(
            self.adjusted_time_RV,
            self.obs_RV_adjusted,
            yerr=self.err_RV,
            fmt="o",
            label="Observations",
            color="blue",
            alpha=0.8,
        )
        axs[0, 0].plot(self.time_dense, self.mu, label="GP Model", color="red")
        axs[0, 0].fill_between(
            self.time_dense,
            self.mu - self.std,
            self.mu + self.std,
            color="red",
            alpha=0.3,
            label="Confidence Interval",
        )
        axs[0, 0].set_ylabel("RV [m/s]", fontsize=14)

        self.residuals = self.obs_RV_adjusted - self.mu_t
        axs[1, 0].errorbar(
            self.adjusted_time_RV,
            self.residuals,
            yerr=self.err_RV,
            fmt="o",
            color="red",
            label="Residuals",
        )
        axs[1, 0].axhline(0, linestyle="--", color="black")
        axs[1, 0].set_xlabel("Time [BJD - 2457000]", fontsize=14)
        axs[1, 0].set_ylabel("Residuals [m/s]", fontsize=14)

        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_periodograms(self):
        """
        Plot periodograms for observed radial velocities, Gaussian Process predictions, and residuals.

        This method computes Lomb-Scargle periodograms to analyze the frequency content of the data and model predictions,
        highlighting significant periodicities against known orbital periods and assessing the noise floor via false alarm
        probability levels.

        Parameters
        ----------
        None

        Raises
        ------
        RuntimeError
            If an error occurs during the computation of the periodograms, potentially due to data issues or computational limits.

        Notes
        -----
        The method utilizes false alarm probabilities to indicate the significance of detected frequencies in the periodograms.
        """
        min_frequency = 0.002025227467386652
        max_frequency = 0.5177264355163222
        samples_per_peak = 3000
        true_periods = [2.2531136, 3.6906777, 7.4507245, 78.22, 78.22 / 2]
        period_colors = ["red", "green", "blue", "black", "black"]

        periodograms = {}

        try:
            rv_freq, rv_power = LombScargle(
                self.adjusted_time_RV, self.obs_RV_adjusted
            ).autopower(
                minimum_frequency=min_frequency,
                maximum_frequency=max_frequency,
                normalization="standard",
                samples_per_peak=samples_per_peak,
            )
            fap_levels = [0.001, 0.01, 0.1]
            fap_thresholds = LombScargle(
                self.adjusted_time_RV, self.obs_RV_adjusted
            ).false_alarm_level(fap_levels)
            periodograms["RV"] = (rv_freq, rv_power)

            # Periodogram for Gaussian Process predictions
            gp_freq, gp_power = LombScargle(self.adjusted_time_RV, self.mu_t).autopower(
                minimum_frequency=min_frequency,
                maximum_frequency=max_frequency,
                normalization="standard",
                samples_per_peak=samples_per_peak,
            )
            periodograms["RV GP"] = (gp_freq, gp_power)

            # Periodogram for residuals
            res_frequency, res_power = LombScargle(
                self.adjusted_time_RV, self.residuals
            ).autopower(
                minimum_frequency=min_frequency,
                maximum_frequency=max_frequency,
                normalization="standard",
                samples_per_peak=samples_per_peak,
            )
            periodograms["Residuals"] = (res_frequency, res_power)

            # Window function to assess sampling effects
            observation_indicator = np.ones_like(self.adjusted_time_RV)
            window_frequency, window_power = LombScargle(
                self.adjusted_time_RV, observation_indicator
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
        fig, axes = plt.subplots(len(periodograms), 1, figsize=(17.5, 10), sharex=True)
        for i, (ax, (key, (frequency, power))) in enumerate(
            zip(axes, periodograms.items())
        ):
            frq = convert_to_microhertz(frequency)
            ax.plot(frq, power, color="black", label=key)
            ax.axhline(
                fap_thresholds[0], color="black", linestyle="dotted", label="0.1% FAP"
            )
            ax.axhline(
                fap_thresholds[1], color="black", linestyle="dashdot", label="1% FAP"
            )
            ax.axhline(
                fap_thresholds[2], color="black", linestyle="dashed", label="10% FAP"
            )
            for peak, color in zip(true_periods, period_colors):
                ax.axvline(
                    x=convert_to_microhertz(1 / peak),
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                )
            ax.legend(loc="upper right", fontsize=18, handletextpad=1.0, handlelength=0)
            ax.set_ylabel(f"{key} Power", fontsize=20)

        fig.supxlabel("Frequency [µHz]", fontsize=20)
        fig.supylabel("Normalised Power", fontsize=20)
        plt.tight_layout()
        plt.show()

    def plot_phase_folded_rv(self, num_bins=20):
        """
        Plots phase-folded radial velocity data using Gaussian Process regression.

        This method takes phase-folded radial velocity data, fits a Gaussian Process model,
        and plots both the raw and binned data alongside the GP fit.

        Parameters
        ----------
        num_bins : int
            Number of bins used for averaging the phase-folded data.

        Raises
        ------
        RuntimeError
            If there is an error in computing the Gaussian Process or during plotting.
        """
        try:
            # Parameters for planet c, could be modified to be method parameters
            Pc = 3.690677757166691  # Orbital period in days
            log_period = np.log(Pc)
            A_RV = 8.089612560529813
            gamma = 0.4605994875623325
            t_decay = 404.83874800695094

            # Construct the kernel for Gaussian Process
            K_trial_c = (
                A_RV
                * kernels.ExpSine2Kernel(gamma=gamma, log_period=log_period)
                * kernels.ExpSquaredKernel(metric=t_decay)
            )

            # Calculate phases for each observation
            times = np.array(self.adjusted_time_RV)
            phases_c = (times % Pc) / Pc
            sorted_indices_c = np.argsort(phases_c)

            # Sort times, RV data, and errors according to calculated phases
            rv_data = np.array(self.residuals)
            rv_errors = np.array(self.err_RV)
            phases_sorted_c = phases_c[sorted_indices_c]
            rv_sorted_c = rv_data[sorted_indices_c]
            errors_sorted_c = rv_errors[sorted_indices_c]

            # Set up and compute the Gaussian Process
            gp_c = GP(K_trial_c)
            gp_c.compute(times, rv_errors)

            # Predict the GP model over a fine grid for smooth plotting
            fine_phases_c = np.linspace(0, 1, 500)
            fine_times_c = times[0] + fine_phases_c * Pc
            mu_fit_c, var_fit_c = gp_c.predict(
                rv_sorted_c, fine_times_c, return_var=True
            )
            std_fit_c = np.sqrt(var_fit_c)

            # Binning the data for clarity
            bins_c = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(phases_sorted_c, bins_c) - 1
            binned_rv_c = np.zeros(num_bins)
            binned_errors_c = np.zeros(num_bins)
            binned_phases_c = np.zeros(num_bins)

            for i in range(num_bins):
                in_bin = bin_indices == i
                if np.any(in_bin):
                    binned_rv_c[i] = np.mean(rv_sorted_c[in_bin])
                    binned_errors_c[i] = np.std(rv_sorted_c[in_bin]) / np.sqrt(
                        np.sum(in_bin)
                    )
                    binned_phases_c[i] = np.mean(phases_sorted_c[in_bin])

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                phases_sorted_c,
                rv_sorted_c,
                yerr=errors_sorted_c,
                fmt=".",
                label="Observed Data",
                color="k",
                alpha=0.6,
            )
            plt.errorbar(
                binned_phases_c,
                binned_rv_c,
                yerr=binned_errors_c,
                fmt="o",
                label="Binned Observed Data",
                color="blue",
                alpha=0.8,
            )
            plt.plot(
                fine_phases_c, mu_fit_c, color="r", label="GP Model Fit", linewidth=2
            )
            plt.xlabel("Phase")
            plt.ylabel("Radial Velocity (m/s)")
            plt.title("Phase-folded RV Data with GP Model Fit")
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            raise RuntimeError(f"Error in plot_phase_folded_rv: {str(e)}")


if __name__ == "__main__":
    plot_utils = PlotUtils("datasets/cleaned_data_20240531.pickle")
    plot_utils.plot_gp()
    plot_utils.plot_periodograms()
    plot_utils.plot_phase_folded_rv()
