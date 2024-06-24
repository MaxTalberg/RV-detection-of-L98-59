# --- Imports
import os
import time
import copy
import pickle
import radvel
import george
import anesthetic
import pypolychord

import numpy as np
import pandas as pd
import getdist.plots as gdplots
from getdist import MCSamples
from george import kernels
from anesthetic import read_chains, make_1d_axes, make_2d_axes
from pypolychord.settings import PolyChordSettings

import prior_transforms as pt
from pickle_data import unpickle_data


class L9859Analysis:

    def __init__(
        self,
        filepath,
        include_planet_b,
        include_fwhm,
        include_sindex,
        polychord_settings=None,
        output_params=None,
    ):

        self.filepath = filepath
        self.include_planet_b = include_planet_b
        self.include_fwhm = include_fwhm
        self.include_sindex = include_sindex
        self.polychord_settings = (
            polychord_settings
            if polychord_settings
            else {
                "do_clustering": True,
                "precision_criterion": 0.01,
                "num_repeats": 1,
                "read_resume": False,
                "nprior": 5000,
                "nfail": 50000,
            }
        )
        self.output_params = (
            output_params
            if output_params
            else {"base_dir": "output_dir/specific_run", "feedback": 1}
        )
        self.output_directory = self.output_params["base_dir"]
        self.e_b = self.e_c = self.e_d = None
        self.load_data()
        self.initialise_parameters()
        self.create_qp_kernel()

    def load_data(self):
        """
        Loads and preprocesses the data from the specified filepath.

        Processes the CSV file to extract time, radial velocity, FWHM, and BIS data along with their errors.
        Sets up derived parameters for the data such as maximum jitter and statistical measures (median, std, peak-to-peak).
        """
        X = unpickle_data(self.filepath)
        self.X_pre, self.X_post, self.X_harps = (
            X["ESPRESSO_pre"],
            X["ESPRESSO_post"],
            X["HARPS"],
        )
        self.n_pre, self.n_post, self.n_harps = (
            len(self.X_pre["RV"]),
            len(self.X_post["RV"]),
            len(self.X_harps["RV"]),
        )

        # --- Scale, offset and jitter
        self.rv_err_pre, self.rv_err_post, self.rv_err_harps = (
            self.X_pre["e_RV"],
            self.X_post["e_RV"],
            self.X_harps["e_RV"],
        )

        # Combine all RV data
        self.time_RV = np.block(
            [self.X_pre["Time"], self.X_post["Time"], self.X_harps["Time"]]
        )
        self.obs_RV = np.block(
            [self.X_pre["RV"], self.X_post["RV"], self.X_harps["RV"]]
        )
        self.adjusted_time_RV = self.time_RV - 2457000

        if self.include_fwhm:
            self.fwhm_obs_pre, self.fwhm_obs_post = (
                self.X_pre["FWHM"] / 1000,
                self.X_post["FWHM"] / 1000,
            )  # Convert to km/s
            self.fwhm_err_pre, self.fwhm_err_post = (
                self.X_pre["e_FWHM"],
                self.X_post["e_FWHM"],
            )

            # Combine all FWHM data
            self.obs_FWHM = np.block(
                [self.X_pre["FWHM"] / 1000, self.X_post["FWHM"] / 1000]
            )
            self.time_FWHM = np.block([self.X_pre["Time"], self.X_post["Time"]])
            self.adjusted_time_FWHM = self.time_FWHM - 2457000

        if self.include_sindex:
            self.sindex_obs_pre, self.sindex_obs_post, self.sindex_obs_harps = (
                self.X_pre["Sindex"],
                self.X_post["Sindex"],
                self.X_harps["Sindex"],
            )
            self.sindex_err_pre, self.sindex_err_post, self.sindex_err_harps = (
                self.X_pre["e_Sindex"],
                self.X_post["e_Sindex"],
                self.X_harps["e_Sindex"],
            )

            self.obs_Sindex = np.block(
                [self.X_pre["Sindex"], self.X_post["Sindex"], self.X_harps["Sindex"]]
            )

    def initialise_parameters(self):
        params_general = [
            "A_RV",
            "P_rot",
            "t_decay",
            "gamma",
            "sigma_RV_pre",
            "sigma_RV_post",
            "sigma_RV_harps",
            "v0_pre",
            "off_post",
            "off_harps",
        ]

        params_general_latex = [
            r"A_{RV}",
            r"P_{rot}",
            r"lambda_p",
            r"lambda_e",
            r"sigma_{RV,pre}",
            r"sigma_{RV,post}",
            r"sigma_{RV,HARPS}",
            r"v_{0,pre}",
            r"off_{post}",
            r"off_{HARPS}",
        ]

        params_fwhm = [
            "A_FWHM",
            "C_FWHM_pre",
            "C_FWHM_post",
            "sigma_FWHM_pre",
            "sigma_FWHM_post",
        ]
        params_fwhm_latex = [
            r"A_{FWHM}",
            r"C_{FWHM,pre}",
            r"C_{FWHM,post}",
            r"sigma_{FWHM,pre}",
            r"sigma_{FWHM,post}",
        ]

        params_sindex = [
            "A_Sindex",
            "C_Sindex_pre",
            "C_Sindex_post",
            "C_Sindex_harps",
            "sigma_Sindex_pre",
            "sigma_Sindex_post",
            "sigma_Sindex_harps",
        ]
        params_sindex_latex = [
            r"A_{Sindex}",
            r"C_{Sindex,pre}",
            r"C_{Sindex,post}",
            r"C_{Sindex,HARPS}",
            r"sigma_{Sindex,pre}",
            r"sigma_{Sindex,post}",
            r"sigma_{Sindex,HARPS}",
        ]

        params_planet_b = ["P_b", "Tc_b", "secosw_b", "sesinw_b", "K_b", "w_b"]
        params_planet_c = ["P_c", "Tc_c", "secosw_c", "sesinw_c", "K_c", "w_c"]
        params_planet_d = ["P_d", "Tc_d", "secosw_d", "sesinw_d", "K_d", "w_d"]

        params_derived_b = ["e_b*"]
        params_derived_c = ["e_c*"]
        params_derived_d = ["e_d*"]

        if self.include_planet_b:
            planet_params = params_planet_b + params_planet_c + params_planet_d
            self.derived_params = params_derived_b + params_derived_c + params_derived_d

        else:
            planet_params = params_planet_c + params_planet_d
            self.derived_params = params_derived_c + params_derived_d

        planet_params_latex = copy.deepcopy(planet_params)
        derived_params_latex = copy.deepcopy(self.derived_params)

        if self.include_fwhm:
            params_general += params_fwhm
            params_general_latex += params_fwhm_latex

        if self.include_sindex:
            params_general += params_sindex
            params_general_latex += params_sindex_latex

        self.parameters = params_general + planet_params
        self.parameters_latex = (
            params_general_latex + planet_params_latex + derived_params_latex
        )

        self.Q = {self.parameters[i]: i for i in range(len(self.parameters))}
        self.nDims = len(self.parameters)
        self.nDerived = len(self.derived_params)

    def create_qp_kernel(self):

        amplitude = 2.44
        gamma = 3.2
        log_period = np.log(78)
        length_scale = 49

        periodic_kernel = amplitude * kernels.ExpSine2Kernel(
            gamma=gamma, log_period=log_period
        )
        decay_kernel = kernels.ExpSquaredKernel(metric=length_scale)
        self.qp_kernel = periodic_kernel * decay_kernel

        self.gp = george.GP(self.qp_kernel)
        self.gp_fwhm = None
        self.gp_sindex = None
        if self.include_fwhm:
            self.gp_fwhm = george.GP(self.qp_kernel)
        if self.include_sindex:
            self.gp_sindex = george.GP(self.qp_kernel)

    def derive_params(self, q):

        # --- Derived parameters

        # planet b
        if self.include_planet_b:
            e_b = np.sqrt(q[self.Q["secosw_b"]] ** 2 + q[self.Q["sesinw_b"]] ** 2)
            self.e_b = e_b

        # planet c
        e_c = np.sqrt(q[self.Q["secosw_c"]] ** 2 + q[self.Q["sesinw_c"]] ** 2)

        # planet d
        e_d = np.sqrt(q[self.Q["secosw_d"]] ** 2 + q[self.Q["sesinw_d"]] ** 2)

        self.e_c = e_c
        self.e_d = e_d

    def compute_planets_RV(self, T0, q):

        RV_total = np.zeros(len(T0))
        self.derive_params(q)

        if self.include_planet_b:
            # Initialize RadVel parameters with correct mapping
            radvel_params = radvel.Parameters(
                3, basis="per tc e w k", planet_letters={1: "b", 2: "c", 3: "d"}
            )

            # planet b
            radvel_params["per1"] = radvel.Parameter(value=q[self.Q["P_b"]])
            radvel_params["tc1"] = radvel.Parameter(value=q[self.Q["Tc_b"]])
            radvel_params["e1"] = radvel.Parameter(value=self.e_b)
            radvel_params["w1"] = radvel.Parameter(value=q[self.Q["w_b"]])
            radvel_params["k1"] = radvel.Parameter(value=q[self.Q["K_b"]])

            # planet c
            radvel_params["per2"] = radvel.Parameter(value=q[self.Q["P_c"]])
            radvel_params["tc2"] = radvel.Parameter(value=q[self.Q["Tc_c"]])
            radvel_params["e2"] = radvel.Parameter(value=self.e_c)
            radvel_params["w2"] = radvel.Parameter(value=q[self.Q["w_c"]])
            radvel_params["k2"] = radvel.Parameter(value=q[self.Q["K_c"]])

            # planet d
            radvel_params["per3"] = radvel.Parameter(value=q[self.Q["P_d"]])
            radvel_params["tc3"] = radvel.Parameter(value=q[self.Q["Tc_d"]])
            radvel_params["e3"] = radvel.Parameter(value=self.e_d)
            radvel_params["w3"] = radvel.Parameter(value=q[self.Q["w_d"]])
            radvel_params["k3"] = radvel.Parameter(value=q[self.Q["K_d"]])

        else:

            # Initialize RadVel parameters with correct mapping
            radvel_params = radvel.Parameters(
                2, basis="per tc e w k", planet_letters={1: "c", 2: "d"}
            )

            # planet c
            radvel_params["per1"] = radvel.Parameter(value=q[self.Q["P_c"]])
            radvel_params["tc1"] = radvel.Parameter(value=q[self.Q["Tc_c"]])
            radvel_params["e1"] = radvel.Parameter(value=self.e_c)
            radvel_params["w1"] = radvel.Parameter(value=q[self.Q["w_c"]])
            radvel_params["k1"] = radvel.Parameter(value=q[self.Q["K_c"]])

            # planet d
            radvel_params["per2"] = radvel.Parameter(value=q[self.Q["P_d"]])
            radvel_params["tc2"] = radvel.Parameter(value=q[self.Q["Tc_d"]])
            radvel_params["e2"] = radvel.Parameter(value=self.e_d)
            radvel_params["w2"] = radvel.Parameter(value=q[self.Q["w_d"]])
            radvel_params["k2"] = radvel.Parameter(value=q[self.Q["K_d"]])

        # Make sure to use a model setup correctly
        model = radvel.RVModel(radvel_params)

        RV_total += model(T0)

        return RV_total

    def planet_prior(self, qq):
        if self.include_planet_b:
            # planet b
            qq[self.Q["P_b"]] = pt.gaussian(qq[self.Q["P_b"]], 2.2531136, 1.5e-6)
            qq[self.Q["Tc_b"]] = pt.gaussian(qq[self.Q["Tc_b"]], 1366.1708, 3e-4)
            qq[self.Q["secosw_b"]] = pt.kipping_beta(qq[self.Q["secosw_b"]])
            qq[self.Q["sesinw_b"]] = pt.kipping_beta(qq[self.Q["sesinw_b"]])
            qq[self.Q["K_b"]] = pt.uniform(qq[self.Q["K_b"]], 0, 17)
            qq[self.Q["w_b"]] = pt.uniform(qq[self.Q["w_b"]], -np.pi, np.pi)

        # planet c
        qq[self.Q["P_c"]] = pt.gaussian(qq[self.Q["P_c"]], 3.6906777, 2.6e-6)
        qq[self.Q["Tc_c"]] = pt.gaussian(qq[self.Q["Tc_c"]], 1367.2751, 6e-4)
        qq[self.Q["secosw_c"]] = pt.kipping_beta(qq[self.Q["secosw_c"]])
        qq[self.Q["sesinw_c"]] = pt.kipping_beta(qq[self.Q["sesinw_c"]])
        qq[self.Q["K_c"]] = pt.uniform(qq[self.Q["K_c"]], 0, 17)
        qq[self.Q["w_c"]] = pt.uniform(qq[self.Q["w_c"]], -np.pi, np.pi)

        # planet d
        qq[self.Q["P_d"]] = pt.gaussian(qq[self.Q["P_d"]], 7.4507245, 8.1e-6)
        qq[self.Q["Tc_d"]] = pt.gaussian(qq[self.Q["Tc_d"]], 1362.7375, 8e-4)
        qq[self.Q["secosw_d"]] = pt.kipping_beta(qq[self.Q["secosw_d"]])
        qq[self.Q["sesinw_d"]] = pt.kipping_beta(qq[self.Q["sesinw_d"]])
        qq[self.Q["K_d"]] = pt.uniform(qq[self.Q["K_d"]], 0, 17)
        qq[self.Q["w_d"]] = pt.uniform(qq[self.Q["w_d"]], -np.pi, np.pi)

        return qq

    def myprior(self, q):

        qq = np.copy(q)

        # priors
        qq[self.Q["A_RV"]] = pt.uniform(q[self.Q["A_RV"]], 0, 16.8)  # U(0, 17)
        qq[self.Q["P_rot"]] = pt.jeffreys(q[self.Q["P_rot"]], 5, 520)  # J(5, 520)
        qq[self.Q["t_decay"]] = pt.jeffreys(
            q[self.Q["t_decay"]], qq[self.Q["P_rot"]] / 2, 2600
        )  # T_decay > P_rot/2 + J(2.5, 2600)
        qq[self.Q["gamma"]] = pt.uniform(q[self.Q["gamma"]], 0.05, 5)  # U(0.05, 5)
        qq[self.Q["sigma_RV_pre"]] = pt.uniform(
            q[self.Q["sigma_RV_pre"]], 0, 3.97059
        )  # U(0, max_jitter_pre)
        qq[self.Q["sigma_RV_post"]] = pt.uniform(
            q[self.Q["sigma_RV_post"]], 0, 3.28532
        )  # U(0, max_jitter_post)
        qq[self.Q["sigma_RV_harps"]] = pt.uniform(q[self.Q["sigma_RV_harps"]], 0, 10.5)
        qq[self.Q["v0_pre"]] = pt.gaussian(
            q[self.Q["v0_pre"]], -5579.2, 35
        )  # N(-5579.1, 35)
        qq[self.Q["off_post"]] = pt.gaussian(
            q[self.Q["off_post"]], 2.86, 4.65
        )  # N(2.88, 4.8)
        qq[self.Q["off_harps"]] = pt.gaussian(
            q[self.Q["off_harps"]], -99.4, 4.9
        )  # N(-99.5, 5.0)

        if self.include_fwhm:
            qq[self.Q["A_FWHM"]] = pt.uniform(q[self.Q["A_FWHM"]], 0, 0.03755)
            qq[self.Q["C_FWHM_pre"]] = pt.gaussian(
                q[self.Q["C_FWHM_pre"]], 4.50518, 0.0086
            )
            qq[self.Q["C_FWHM_post"]] = pt.gaussian(
                q[self.Q["C_FWHM_post"]], 4.5169, 0.0103
            )
            qq[self.Q["sigma_FWHM_pre"]] = pt.uniform(
                q[self.Q["sigma_FWHM_pre"]], 0, 7.941
            )
            qq[self.Q["sigma_FWHM_post"]] = pt.uniform(
                q[self.Q["sigma_FWHM_post"]], 0, 6.571
            )

        if self.include_sindex:
            qq[self.Q["A_Sindex"]] = pt.uniform(q[self.Q["A_Sindex"]], 0, 1.25)
            qq[self.Q["C_Sindex_pre"]] = pt.gaussian(
                q[self.Q["C_Sindex_pre"]], 0.633095, 0.09636
            )
            qq[self.Q["C_Sindex_post"]] = pt.gaussian(
                q[self.Q["C_Sindex_post"]], 0.685562, 0.0769
            )
            qq[self.Q["C_Sindex_harps"]] = pt.gaussian(
                q[self.Q["C_Sindex_harps"]], 0.72, 0.14663
            )
            qq[self.Q["sigma_Sindex_pre"]] = pt.uniform(
                q[self.Q["sigma_Sindex_pre"]], 0, 0.0127
            )
            qq[self.Q["sigma_Sindex_post"]] = pt.uniform(
                q[self.Q["sigma_Sindex_post"]], 0, 0.010445
            )

            qq[self.Q["sigma_Sindex_harps"]] = pt.uniform(
                q[self.Q["sigma_Sindex_harps"]], 0, 0.700
            )

        # planet priors
        qq = self.planet_prior(qq)

        return qq

    def mean_fxn(self, T0, q):
        Y0 = np.zeros(T0.shape)
        Y1 = np.zeros(self.adjusted_time_FWHM.shape) if self.include_fwhm else None
        Y2 = np.zeros(T0.shape) if self.include_sindex else None

        # ESPRESSO pre offset
        Y0[0 : self.n_pre] += q[self.Q["v0_pre"]]
        # ESPRESSO post offset
        Y0[self.n_pre : self.n_pre + self.n_post] += (
            q[self.Q["v0_pre"]] + q[self.Q["off_post"]]
        )
        # HARPS offset
        Y0[self.n_pre + self.n_post :] += q[self.Q["v0_pre"]] + q[self.Q["off_harps"]]

        if self.include_fwhm:
            # ESPRESSO pre offset
            Y1[0 : self.n_pre] += q[self.Q["C_FWHM_pre"]]
            # ESPRESSO post offset
            Y1[self.n_pre : self.n_pre + self.n_post] += q[self.Q["C_FWHM_post"]]

        if self.include_sindex:
            # ESPRESSO pre offset
            Y2[0 : self.n_pre] += q[self.Q["C_Sindex_pre"]]
            # ESPRESSO post offset
            Y2[self.n_pre : self.n_pre + self.n_post] += q[self.Q["C_Sindex_post"]]
            # HARPS offset
            Y2[self.n_pre + self.n_post :] += q[self.Q["C_Sindex_harps"]]

        # Compute the RV model with corrected offsets
        Y0 += self.compute_planets_RV(T0, q)

        results = [Y0, None, None]  # RV results are always included

        if self.include_fwhm:
            results[1] = Y1
        if self.include_sindex:
            results[2] = Y2

        return results

    def myloglike(self, theta):

        q = np.copy(theta)

        # Derived parameters
        self.derive_params(q)
        dps = (
            [self.e_b, self.e_c, self.e_d]
            if self.include_planet_b
            else [self.e_c, self.e_d]
        )

        A_RV = q[self.Q["A_RV"]]
        gamma = q[self.Q["gamma"]]
        log_period = np.log(q[self.Q["P_rot"]])
        t_decay = q[self.Q["t_decay"]]

        p0 = np.array([A_RV, gamma, log_period, t_decay])
        self.gp.set_parameter_vector(p0)

        # Compute the RV error
        err_RV = np.block(
            [
                np.sqrt(self.rv_err_pre**2 + q[self.Q["sigma_RV_pre"]] ** 2),
                np.sqrt(self.rv_err_post**2 + q[self.Q["sigma_RV_post"]] ** 2),
                np.sqrt(self.rv_err_harps**2 + q[self.Q["sigma_RV_harps"]] ** 2),
            ]
        )

        # Compute the GP model
        self.gp.compute(self.adjusted_time_RV, err_RV)

        # Compute the RV model and residuals
        mu_RV, mu_FWHM, mu_Sindex = self.mean_fxn(self.adjusted_time_RV, q)
        residuals_RV = self.obs_RV - mu_RV

        # Compute the log likelihood
        log_likelihood = self.gp.log_likelihood(residuals_RV)

        if self.include_fwhm:
            # GP parameters for FWHM
            A_FWHM = q[self.Q["A_FWHM"]]
            p1 = np.array([A_FWHM, gamma, log_period, t_decay])
            self.gp_fwhm.set_parameter_vector(p1)

            # Compute the FWHM error
            err_FWHM = np.block(
                [
                    np.sqrt(self.fwhm_err_pre**2 + q[self.Q["sigma_FWHM_pre"]] ** 2),
                    np.sqrt(self.fwhm_err_post**2 + q[self.Q["sigma_FWHM_post"]] ** 2),
                ]
            )
            # Compute the GP model
            self.gp_fwhm.compute(self.adjusted_time_FWHM, err_FWHM)

            # Compute the FWHM model and residuals
            residuals_FWHM = self.obs_FWHM - mu_FWHM
            log_likelihood += self.gp_fwhm.log_likelihood(residuals_FWHM)

        if self.include_sindex:
            # GP parameters for S-index
            A_Sindex = q[self.Q["A_Sindex"]]
            p2 = np.array([A_Sindex, gamma, log_period, t_decay])
            self.gp_sindex.set_parameter_vector(p2)

            # Compute the S-index error
            err_Sindex = np.block(
                [
                    np.sqrt(
                        self.sindex_err_pre**2 + q[self.Q["sigma_Sindex_pre"]] ** 2
                    ),
                    np.sqrt(
                        self.sindex_err_post**2 + q[self.Q["sigma_Sindex_post"]] ** 2
                    ),
                    np.sqrt(
                        self.sindex_err_harps**2 + q[self.Q["sigma_Sindex_harps"]] ** 2
                    ),
                ]
            )
            # Compute the GP model
            self.gp_sindex.compute(self.adjusted_time_RV, err_Sindex)

            # Compute the S-index model and residuals
            residuals_Sindex = self.obs_Sindex - mu_Sindex
            log_likelihood += self.gp_sindex.log_likelihood(residuals_Sindex)

        return log_likelihood, dps

    def dumper(self, live, dead, logweights, logZ, logZerr):
        print("Last dead point:", dead[-1])

    def run_analysis(self):

        settings = PolyChordSettings(
            self.nDims,
            nDerived=self.nDerived,
            **self.polychord_settings,
            **self.output_params,
        )

        t_start = time.time()
        output = pypolychord.run_polychord(
            loglikelihood=self.myloglike,
            nDims=self.nDims,
            nDerived=self.nDerived,
            prior=self.myprior,
            dumper=self.dumper,
            settings=settings,
        )
        t_end = time.time()
        self.runtime = t_end - t_start

    def load_samples_from_file(self, filename):

        file_path = os.path.join(self.output_directory, filename)
        samples = np.loadtxt(file_path)
        return samples

    def handle_results(
        self, file_name="test_equal_weights.txt", plot_name="triangle_plot.pdf"
    ):
        plot_name = "traingle_planet_b_{}_fwhm_{}_sindex_{}.pdf".format(
            self.include_planet_b, self.include_fwhm, self.include_sindex
        )

        samples_file = self.load_samples_from_file(file_name)
        samples_data = samples_file

        first_two_columns = ["log_likelihood", "derived_1"]

        param_names = first_two_columns + self.parameters_latex

        posterior = MCSamples(samples=samples_data, names=param_names)

        means = posterior.getMeans()

        vars = posterior.getVars()
        sds = np.sqrt(vars)

        for index, (name, mean, sd) in enumerate(zip(param_names[1:], means, sds)):
            print(rf"{index}. {name}: {mean} $\pm$ {sd}")

        g = gdplots.get_subplot_plotter()
        g.triangle_plot(posterior, filled=True)
        plot_path = os.path.join(self.output_directory, plot_name)
        g.export(plot_path)

        algo_output = [
            {
                "param_names": param_names,
                "means": means,
                "vars": vars,
                "sds": sds,
                "posterior": posterior,
                "polychord_settings": self.polychord_settings,
            }
        ]

        pickle_file_name = "pickle_planet_b_{}_fwhm_{}_sindex_{}.pickle".format(
            self.include_planet_b, self.include_fwhm, self.include_sindex
        )
        pickle_file_name = os.path.join(self.output_directory, pickle_file_name)

        if self.runtime > 25:
            with open(pickle_file_name, "wb") as f:
                pickle.dump(algo_output, f)
