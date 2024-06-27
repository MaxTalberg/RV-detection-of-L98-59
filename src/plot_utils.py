import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from george import kernels, GP
from astropy.stats import LombScargle

from pickle_data import unpickle_data
from periodograms import convert_to_microhertz


class PlotUtils:
    def __init__(self, pickle_file_path):
        self.pickle_file_path = pickle_file_path
        self.load_data()
        self.setup_gp()

    def load_data(self):
        # Load data from a pickle file
        X = unpickle_data(self.pickle_file_path)
        X_pre, X_post, X_harps = X["ESPRESSO_pre"], X["ESPRESSO_post"], X["HARPS"]
        self.time_pre, self.time_post, self.time_harps = (
            X_pre["Time"],
            X_post["Time"],
            X_harps["Time"],
        )
        self.obs_pre, self.obs_post, self.obs_harps = (
            X_pre["RV"],
            X_post["RV"],
            X_harps["RV"],
        )
        err_pre, err_post, err_harps = X_pre["e_RV"], X_post["e_RV"], X_harps["e_RV"]

        # Calculate adjusted times and observed RVs after applying offsets
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

        # Adjust errors
        self.err_RV = np.block(
            [
                np.sqrt(err_pre**2 + 1.2702928849665707**2),
                np.sqrt(err_post**2 + 0.5959691795036054**2),
                np.sqrt(err_harps**2 + 0.27246571959879184**2),
            ]
        )

    def setup_gp(self):
        log_period = np.log(27.38068137156724)
        self.A_RV = 7.711836893488454
        self.gamma = 2.917125126168333
        self.t_decay = 382.801299740658
        K_trial = (
            self.A_RV
            * kernels.ExpSine2Kernel(gamma=self.gamma, log_period=log_period)
            * kernels.ExpSquaredKernel(metric=self.t_decay)
        )
        self.gp = GP(K_trial)
        self.gp.compute(self.adjusted_time_RV, self.err_RV)
        self.time_dense = np.linspace(
            min(self.adjusted_time_RV), max(self.adjusted_time_RV), 1000
        )
        self.mu, self.var = self.gp.predict(
            self.obs_RV_adjusted, self.time_dense, return_var=True
        )
        self.std = np.sqrt(self.var)
        self.mu_t, self.var_t = self.gp.predict(
            self.obs_RV_adjusted, self.adjusted_time_RV, return_var=True
        )
        self.std_t = np.sqrt(self.var_t)

    def plot_gp(self):
        # Plotting logic for GP results
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

            gp_freq, gp_power = LombScargle(self.adjusted_time_RV, self.mu_t).autopower(
                minimum_frequency=min_frequency,
                maximum_frequency=max_frequency,
                normalization="standard",
                samples_per_peak=samples_per_peak,
            )
            periodograms["RV GP"] = (gp_freq, gp_power)

            res_frequency, res_power = LombScargle(
                self.adjusted_time_RV, self.residuals
            ).autopower(
                minimum_frequency=min_frequency,
                maximum_frequency=max_frequency,
                normalization="standard",
                samples_per_peak=samples_per_peak,
            )
            fap_levels = [0.001, 0.01, 0.1]
            res_fap_thresholds = LombScargle(
                self.adjusted_time_RV, self.residuals
            ).false_alarm_level(fap_levels)
            periodograms["Residuals"] = (res_frequency, res_power)

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
            fap_01 = fap_thresholds[0]  # 0.1% FAP level
            fap_1 = fap_thresholds[1]  # 1% FAP level
            fap_10 = fap_thresholds[2]  # 10% FAP level
            frq = convert_to_microhertz(frequency)
            ax.plot(frq, power, color="black", label=key)
            # ax.set_ylabel(key, fontsize=20)
            if i < len(axes) - 1:
                for peak, color in zip(true_periods, period_colors):
                    ax.axvline(
                        x=convert_to_microhertz(1 / peak),
                        color=color,
                        linestyle="--",
                        linewidth=1.2,
                    )
                ax.legend(
                    loc="upper right", fontsize=18, handletextpad=1.0, handlelength=0
                )

            if i < len(axes) - 1:
                ax.axhline(fap_01, color="black", linestyle="dotted", label="0.1% FAP")
                ax.axhline(fap_1, color="black", linestyle="dashdot", label="1% FAP")
                ax.axhline(fap_10, color="black", linestyle="dashed", label="10% FAP")
            else:
                ax.legend(
                    loc="upper right", fontsize=18, handletextpad=1.0, handlelength=0
                )

            if i == 0:
                ax.text(
                    -0.1,
                    0.26,
                    r"$P_{rot}$",
                    color="black",
                    verticalalignment="top",
                    size=18,
                )
                ax.text(
                    0.40,
                    0.26,
                    r"$\frac{P_{rot}}{2}$",
                    color="black",
                    verticalalignment="top",
                    size=18,
                )
                ax.text(
                    1.45, 0.255, r"$P_d$", color="b", verticalalignment="top", size=20
                )
                ax.text(
                    3.05, 0.255, r"$P_c$", color="g", verticalalignment="top", size=20
                )
                ax.text(
                    5.04, 0.255, r"$P_b$", color="r", verticalalignment="top", size=20
                )
                ax.text(
                    6.35, 0.155, "0.1%", color="black", verticalalignment="top", size=18
                )
                ax.text(
                    6.35, 0.115, "1%", color="black", verticalalignment="top", size=18
                )
                ax.text(
                    6.35, 0.075, "10%", color="black", verticalalignment="top", size=18
                )

        fig.supxlabel("Frequency [µHz]", fontsize=20)
        fig.supylabel("Normalised Power", fontsize=20)

        plt.tight_layout()
        plt.show()

    def plot_phase_folded_rv(self, num_bins=20):
        Pc = 3.690677757166691  # period
        times = np.array(self.adjusted_time_RV)  # observational times
        rv_data = np.array(self.residuals)  # radial velocity measurements
        rv_errors = np.array(self.err_RV)  # errors in radial velocity measurements

        log_period = np.log(Pc)
        A_RV = 8.089612560529813
        gamma = 0.4605994875623325
        t_decay = 404.83874800695094

        K_trial_c = (
            A_RV
            * kernels.ExpSine2Kernel(gamma=gamma, log_period=log_period)
            * kernels.ExpSquaredKernel(metric=t_decay)
        )

        # Calculate the phases for each observation
        phases_c = (times % Pc) / Pc
        sorted_indices_c = np.argsort(phases_c)  # Sorting indices

        # Sorting the times, RV data, and errors according to phase
        phases_sorted_c = phases_c[sorted_indices_c]
        rv_sorted_c = rv_data[sorted_indices_c]
        errors_sorted_c = rv_errors[sorted_indices_c]

        # Generate a fine grid of phases and corresponding times for one period for plotting
        fine_phases_c = np.linspace(
            0, 1, 500
        )  # Increased number of points for smoothness
        fine_times_c = times[0] + fine_phases_c * Pc

        # Setup the Gaussian Process
        gp_c = GP(K_trial_c)
        gp_c.compute(times, rv_errors)  # Make sure to use observation errors

        # Predict the GP model over the fine grid
        mu_fit_c, var_fit_c = gp_c.predict(
            rv_sorted_c, fine_times_c, return_var=True
        )  # Use sorted RV data
        std_fit_c = np.sqrt(var_fit_c)

        num_bins = 20  # Adjust this to increase or decrease the number of bins

        # Bin edges
        bins_c = np.linspace(0, 1, num_bins + 1)

        # Digitize the phases to find out which bin they belong to
        bin_indices = np.digitize(phases_sorted_c, bins_c) - 1

        # Preparing arrays to hold the binned data
        binned_rv_c = np.zeros(num_bins)
        binned_errors_c = np.zeros(num_bins)
        binned_phases_c = np.zeros(num_bins)

        # Calculate the mean RV and standard error in each bin
        for i in range(num_bins):
            in_bin = bin_indices == i
            if np.any(in_bin):
                binned_rv_c[i] = np.mean(rv_sorted_c[in_bin])
                # Standard error of the mean as the error estimate
                binned_errors_c[i] = np.std(rv_sorted_c[in_bin]) / np.sqrt(
                    np.sum(in_bin)
                )
                binned_phases_c[i] = np.mean(phases_sorted_c[in_bin])

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            binned_phases_c,
            binned_rv_c,
            yerr=binned_errors_c,
            fmt="o",
            label="Binned Observed Data",
            color="blue",
            alpha=0.8,
        )
        plt.errorbar(
            phases_sorted_c,
            rv_sorted_c,
            yerr=errors_sorted_c,
            fmt=".",
            label="Observed Data",
            color="k",
            alpha=0.6,
        )
        plt.plot(fine_phases_c, mu_fit_c, color="r", label="GP Model Fit", linewidth=2)
        plt.xlabel("Phase")
        plt.ylabel("Radial Velocity (m/s)")
        plt.title("Phase-folded RV Data with GP Model Fit")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    plot_utils = PlotUtils("datasets/cleaned_data_20240531.pickle")
    plot_utils.plot_gp()
    plot_utils.plot_periodograms()
    plot_utils.plot_phase_folded_rv()
    # Hypothetical setup (replace with your actual data and GP kernel)
