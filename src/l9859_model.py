# --- Imports
import os
import time
import pickle
import radvel
import george

import pypolychord

import numpy as np
import getdist.plots as gdplots
from getdist import MCSamples
from george import kernels
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
        try:
            X = unpickle_data(self.filepath)
            self.X_pre, self.X_post, self.X_harps = (
                X["ESPRESSO_pre"],
                X["ESPRESSO_post"],
                X["HARPS"],
            )
            self.setup_data()

        except OSError as e:
            print(f"Failed to load data: {e}")
            raise

    def setup_data(self):
        """
        Sets up the data for analysis by combining and adjusting observational data.

        Combines radial velocity (RV), full width at half maximum (FWHM), and S-index data
        from different instruments and time periods. Adjusts time and scales FWHM data.

        Raises
        ------
        KeyError
            If required keys are missing in the input data.
        """
        try:
            # Get the number of data points for each dataset
            self.n_pre, self.n_post, self.n_harps = (
                len(self.X_pre["RV"]),
                len(self.X_post["RV"]),
                len(self.X_harps["RV"]),
            )

            # Set up RV errors for each dataset
            self.rv_err_pre, self.rv_err_post, self.rv_err_harps = (
                self.X_pre["e_RV"],
                self.X_post["e_RV"],
                self.X_harps["e_RV"],
            )

            # Combine all RV data and adjust time
            self.time_RV = np.block(
                [self.X_pre["Time"], self.X_post["Time"], self.X_harps["Time"]]
            )
            self.obs_RV = np.block(
                [self.X_pre["RV"], self.X_post["RV"], self.X_harps["RV"]]
            )
            self.adjusted_time_RV = self.time_RV - 2457000

            if self.include_fwhm:
                # Convert FWHM data to km/s and set up FWHM errors
                self.fwhm_obs_pre, self.fwhm_obs_post = (
                    self.X_pre["FWHM"] / 1000,
                    self.X_post["FWHM"] / 1000,
                )
                self.fwhm_err_pre, self.fwhm_err_post = (
                    self.X_pre["e_FWHM"],
                    self.X_post["e_FWHM"],
                )

                # Combine all FWHM data and adjust time
                self.obs_FWHM = np.block(
                    [self.X_pre["FWHM"] / 1000, self.X_post["FWHM"] / 1000]
                )
                self.time_FWHM = np.block([self.X_pre["Time"], self.X_post["Time"]])
                self.adjusted_time_FWHM = self.time_FWHM - 2457000

            if self.include_sindex:
                # Set up S-index data and errors for each dataset
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

                # Combine all S-index data
                self.obs_Sindex = np.block(
                    [
                        self.X_pre["Sindex"],
                        self.X_post["Sindex"],
                        self.X_harps["Sindex"],
                    ]
                )

        except KeyError as e:
            print(f"Missing key in data setup: {e}")
            raise

    def initialise_parameters(self):
        """
        Initialises the parameter sets and derived parameters for the analysis.

        Sets up the general, FWHM, S-index, and planetary parameters, and constructs
        parameter dictionaries and lists in both plain and LaTeX formats.
        """
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
            r"$A_{RV}$",
            r"$P_{rot}$",
            r"$\lambda_e$",
            r"$\lambda_p$",
            r"$\sigma_{RV,pre}$",
            r"$\sigma_{RV,post}$",
            r"$\sigma_{RV,HARPS}$",
            r"$v0_{pre}$",
            r"$ΔRV_{post/pre}$",
            r"$ΔRV_{HARPS/pre}$",
        ]

        params_fwhm = [
            "A_FWHM",
            "C_FWHM_pre",
            "C_FWHM_post",
            "sigma_FWHM_pre",
            "sigma_FWHM_post",
        ]
        params_fwhm_latex = [
            r"$A_{FWHM}$",
            r"$C_{FWHM,pre}$",
            r"$C_{FWHM,post}$",
            r"$sigma_{FWHM,pre}$",
            r"$Sigma_{FWHM,post}$",
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
            r"$A_{Sindex}$",
            r"$C_{Sindex,pre}$",
            r"$C_{Sindex,post}$",
            r"$C_{Sindex,HARPS}$",
            r"$\sigma_{Sindex,pre}$",
            r"$\sigma_{Sindex,post}$",
            r"$\sigma_{Sindex,HARPS}$",
        ]

        params_planet_b = ["P_b", "Tc_b", "secosw_b", "sesinw_b", "K_b", "w_b"]
        params_planet_b_latex = [
            r"$P_b$",
            r"$Tc_b$",
            r"$e\cos{\omega}_b$",
            r"$e\sin{\omega}_b$",
            r"$K_b$",
            r"$w_b$",
        ]
        params_planet_c = ["P_c", "Tc_c", "secosw_c", "sesinw_c", "K_c", "w_c"]
        params_planet_c_latex = [
            r"$P_c$",
            r"$Tc_c$",
            r"$e\cos{\omega}_c$",
            r"$e\sin{\omega}_c$",
            r"$K_c$",
            r"$w_c$",
        ]
        params_planet_d = ["P_d", "Tc_d", "secosw_d", "sesinw_d", "K_d", "w_d"]
        params_planet_d_latex = [
            r"$P_d$",
            r"$Tc_d$",
            r"$e\cos{\omega}_d$",
            r"$e\sin{\omega}_d$",
            r"$K_d$",
            r"$w_d$",
        ]
        params_derived_b = ["e_b*"]
        params_derived_b_latex = [r"$e_b$"]
        params_derived_c = ["e_c*"]
        params_derived_c_latex = [r"$e_c$"]
        params_derived_d = ["e_d*"]
        params_derived_d_latex = [r"$e_d$"]

        if self.include_planet_b:
            planet_params = params_planet_b + params_planet_c + params_planet_d
            self.derived_params = params_derived_b + params_derived_c + params_derived_d
            planet_params_latex = (
                params_planet_b_latex + params_planet_c_latex + params_planet_d_latex
            )
            derived_params_latex = (
                params_derived_b_latex + params_derived_c_latex + params_derived_d_latex
            )
        else:
            planet_params = params_planet_c + params_planet_d
            self.derived_params = params_derived_c + params_derived_d
            planet_params_latex = params_planet_c_latex + params_planet_d_latex
            derived_params_latex = params_derived_c_latex + params_derived_d_latex

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
        """
        Creates a Quasi-Periodic (QP) kernel for Gaussian Process (GP) modeling.

        This method sets up a QP kernel by combining an exponential sine-squared kernel
        (periodic kernel) with an exponential squared kernel (decay kernel). It then
        initializes GP models for RV, FWHM, and S-index data based on the inclusion flags.

        Raises
        ------
        Exception
            If there is an error in creating or initialising the kernels.
        """
        try:
            # Define kernel parameters
            amplitude = 2.44
            gamma = 3.2
            log_period = np.log(78)
            length_scale = 49

            # Create the periodic and decay kernels
            periodic_kernel = amplitude * kernels.ExpSine2Kernel(
                gamma=gamma, log_period=log_period
            )
            decay_kernel = kernels.ExpSquaredKernel(metric=length_scale)
            self.qp_kernel = periodic_kernel * decay_kernel

            # Initialize the main GP model with the QP kernel
            self.gp = george.GP(self.qp_kernel)
            self.gp_fwhm = None
            self.gp_sindex = None

            # Initialize additional GP models for FWHM and S-index if included
            if self.include_fwhm:
                self.gp_fwhm = george.GP(self.qp_kernel)
            if self.include_sindex:
                self.gp_sindex = george.GP(self.qp_kernel)

        except Exception as e:
            print(f"Error in creating QP kernel or initializing GPs: {e}")
            raise

    def derive_params(self, q):
        """
        Derives eccentricity parameters for the planets from the input parameter vector.

        Parameters
        ----------
        q : array_like
            Input parameter vector containing `secosw` and `sesinw` for planets.

        Raises
        ------
        KeyError
            If required keys are missing in the parameter vector.
        """
        try:
            # Compute derived parameters for planet b if included
            if self.include_planet_b:
                e_b = np.sqrt(q[self.Q["secosw_b"]] ** 2 + q[self.Q["sesinw_b"]] ** 2)
                self.e_b = e_b

            # Compute derived parameters for planet c
            e_c = np.sqrt(q[self.Q["secosw_c"]] ** 2 + q[self.Q["sesinw_c"]] ** 2)

            # Compute derived parameters for planet d
            e_d = np.sqrt(q[self.Q["secosw_d"]] ** 2 + q[self.Q["sesinw_d"]] ** 2)

            # Assign derived parameters to instance variables
            self.e_c = e_c
            self.e_d = e_d

        except KeyError as e:
            print(f"Missing key in parameter vector: {e}")
            raise

    def compute_planets_RV(self, T0, q):
        """
        Computes the total radial velocity (RV) signal from the planets.

        This method calculates the combined RV signal from all included planets using
        the RadVel library, based on the provided time array and parameter vector.

        Parameters
        ----------
        T0 : array_like
            Array of observation times.
        q : array_like
            Input parameter vector containing orbital parameters for the planets.

        Returns
        -------
        RV_total : np.ndarray
            The total RV signal from all included planets.

        Raises
        ------
        KeyError
            If required keys are missing in the parameter vector.
        Exception
            If there is an error in computing the RV model.
        """
        try:
            # Initialize the total RV array
            RV_total = np.zeros(len(T0))

            # Derive parameters from the input vector
            self.derive_params(q)

            if self.include_planet_b:
                # Initialize RadVel parameters for 3 planets (b, c, d)
                radvel_params = radvel.Parameters(
                    3, basis="per tc e w k", planet_letters={1: "b", 2: "c", 3: "d"}
                )

                # Set RadVel parameters for planet b
                radvel_params["per1"] = radvel.Parameter(value=q[self.Q["P_b"]])
                radvel_params["tc1"] = radvel.Parameter(value=q[self.Q["Tc_b"]])
                radvel_params["e1"] = radvel.Parameter(value=self.e_b)
                radvel_params["w1"] = radvel.Parameter(value=q[self.Q["w_b"]])
                radvel_params["k1"] = radvel.Parameter(value=q[self.Q["K_b"]])

                # Set RadVel parameters for planet c
                radvel_params["per2"] = radvel.Parameter(value=q[self.Q["P_c"]])
                radvel_params["tc2"] = radvel.Parameter(value=q[self.Q["Tc_c"]])
                radvel_params["e2"] = radvel.Parameter(value=self.e_c)
                radvel_params["w2"] = radvel.Parameter(value=q[self.Q["w_c"]])
                radvel_params["k2"] = radvel.Parameter(value=q[self.Q["K_c"]])

                # Set RadVel parameters for planet d
                radvel_params["per3"] = radvel.Parameter(value=q[self.Q["P_d"]])
                radvel_params["tc3"] = radvel.Parameter(value=q[self.Q["Tc_d"]])
                radvel_params["e3"] = radvel.Parameter(value=self.e_d)
                radvel_params["w3"] = radvel.Parameter(value=q[self.Q["w_d"]])
                radvel_params["k3"] = radvel.Parameter(value=q[self.Q["K_d"]])

            else:
                # Initialize RadVel parameters for 2 planets (c, d)
                radvel_params = radvel.Parameters(
                    2, basis="per tc e w k", planet_letters={1: "c", 2: "d"}
                )

                # Set RadVel parameters for planet c
                radvel_params["per1"] = radvel.Parameter(value=q[self.Q["P_c"]])
                radvel_params["tc1"] = radvel.Parameter(value=q[self.Q["Tc_c"]])
                radvel_params["e1"] = radvel.Parameter(value=self.e_c)
                radvel_params["w1"] = radvel.Parameter(value=q[self.Q["w_c"]])
                radvel_params["k1"] = radvel.Parameter(value=q[self.Q["K_c"]])

                # Set RadVel parameters for planet d
                radvel_params["per2"] = radvel.Parameter(value=q[self.Q["P_d"]])
                radvel_params["tc2"] = radvel.Parameter(value=q[self.Q["Tc_d"]])
                radvel_params["e2"] = radvel.Parameter(value=self.e_d)
                radvel_params["w2"] = radvel.Parameter(value=q[self.Q["w_d"]])
                radvel_params["k2"] = radvel.Parameter(value=q[self.Q["K_d"]])

            # Initialize RadVel RV model with the defined parameters
            model = radvel.RVModel(radvel_params)

            # Compute the total RV signal by evaluating the model at the observation times
            RV_total += model(T0)

            return RV_total

        except KeyError as e:
            print(f"Missing key in parameter vector: {e}")
            raise
        except Exception as e:
            print(f"Error in computing RV model: {e}")
            raise

    def planet_prior(self, qq):
        """
        Applies prior transformations to the planetary parameters.

        This method adjusts the input parameter vector for the planets using predefined
        prior distributions.

        Parameters
        ----------
        qq : array_like
            Input parameter vector containing planetary parameters.

        Returns
        -------
        qq : array_like
            The parameter vector with prior transformations applied.

        Raises
        ------
        KeyError
            If required keys are missing in the parameter vector.
        Exception
            If there is an error during the prior transformations.
        """
        try:
            if self.include_planet_b:
                # Apply priors for planet b
                qq[self.Q["P_b"]] = pt.gaussian(qq[self.Q["P_b"]], 2.2531136, 1.5e-6)
                qq[self.Q["Tc_b"]] = pt.gaussian(qq[self.Q["Tc_b"]], 1366.1708, 3e-4)
                qq[self.Q["secosw_b"]] = pt.kipping_beta(qq[self.Q["secosw_b"]])
                qq[self.Q["sesinw_b"]] = pt.kipping_beta(qq[self.Q["sesinw_b"]])
                qq[self.Q["K_b"]] = pt.uniform(qq[self.Q["K_b"]], 0, 17)
                qq[self.Q["w_b"]] = pt.uniform(qq[self.Q["w_b"]], -np.pi, np.pi)

            # Apply priors for planet c
            qq[self.Q["P_c"]] = pt.gaussian(qq[self.Q["P_c"]], 3.6906777, 2.6e-6)
            qq[self.Q["Tc_c"]] = pt.gaussian(qq[self.Q["Tc_c"]], 1367.2751, 6e-4)
            qq[self.Q["secosw_c"]] = pt.kipping_beta(qq[self.Q["secosw_c"]])
            qq[self.Q["sesinw_c"]] = pt.kipping_beta(qq[self.Q["sesinw_c"]])
            qq[self.Q["K_c"]] = pt.uniform(qq[self.Q["K_c"]], 0, 17)
            qq[self.Q["w_c"]] = pt.uniform(qq[self.Q["w_c"]], -np.pi, np.pi)

            # Apply priors for planet d
            qq[self.Q["P_d"]] = pt.gaussian(qq[self.Q["P_d"]], 7.4507245, 8.1e-6)
            qq[self.Q["Tc_d"]] = pt.gaussian(qq[self.Q["Tc_d"]], 1362.7375, 8e-4)
            qq[self.Q["secosw_d"]] = pt.kipping_beta(qq[self.Q["secosw_d"]])
            qq[self.Q["sesinw_d"]] = pt.kipping_beta(qq[self.Q["sesinw_d"]])
            qq[self.Q["K_d"]] = pt.uniform(qq[self.Q["K_d"]], 0, 17)
            qq[self.Q["w_d"]] = pt.uniform(qq[self.Q["w_d"]], -np.pi, np.pi)

            return qq

        except KeyError as e:
            print(f"Missing key in parameter vector: {e}")
            raise
        except Exception as e:
            print(f"Error during prior transformations: {e}")
            raise

    def myprior(self, q):
        """
        Applies prior transformations to the input parameter vector.

        This method adjusts the input parameter vector using predefined prior
        distributions for various parameters including RV, FWHM, and S-index parameters.

        Parameters
        ----------
        q : array_like
            Input parameter vector.

        Returns
        -------
        qq : array_like
            The parameter vector with prior transformations applied.

        Raises
        ------
        KeyError
            If required keys are missing in the parameter vector.
        Exception
            If there is an error during the prior transformations.
        """
        try:
            qq = np.copy(q)

            # Apply priors for general parameters
            qq[self.Q["A_RV"]] = pt.uniform(q[self.Q["A_RV"]], 0, 16.8)
            qq[self.Q["P_rot"]] = pt.jeffreys(q[self.Q["P_rot"]], 5, 520)
            qq[self.Q["t_decay"]] = pt.jeffreys(
                q[self.Q["t_decay"]], qq[self.Q["P_rot"]] / 2, 2600
            )
            qq[self.Q["gamma"]] = pt.uniform(q[self.Q["gamma"]], 0.05, 5)
            qq[self.Q["sigma_RV_pre"]] = pt.uniform(
                q[self.Q["sigma_RV_pre"]], 0, 3.97059
            )
            qq[self.Q["sigma_RV_post"]] = pt.uniform(
                q[self.Q["sigma_RV_post"]], 0, 3.28532
            )
            qq[self.Q["sigma_RV_harps"]] = pt.uniform(
                q[self.Q["sigma_RV_harps"]], 0, 10.5
            )
            qq[self.Q["v0_pre"]] = pt.gaussian(q[self.Q["v0_pre"]], -5579.2, 35)
            qq[self.Q["off_post"]] = pt.gaussian(q[self.Q["off_post"]], 2.86, 4.65)
            qq[self.Q["off_harps"]] = pt.gaussian(q[self.Q["off_harps"]], -99.4, 4.9)

            # Apply priors for FWHM parameters if included
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

            # Apply priors for S-index parameters if included
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

            # Apply planet priors
            qq = self.planet_prior(qq)

            return qq

        except KeyError as e:
            print(f"Missing key in parameter vector: {e}")
            raise
        except Exception as e:
            print(f"Error during prior transformations: {e}")
            raise

    def mean_fxn(self, T0, q):
        """
        Computes the mean model for RV, FWHM, and S-index data with appropriate offsets.

        This method calculates the mean model for the radial velocity (RV), full width
        at half maximum (FWHM), and S-index data, applying necessary offsets for different
        observational datasets.

        Parameters
        ----------
        T0 : array_like
            Array of observation times.
        q : array_like
            Input parameter vector.

        Returns
        -------
        results : list
            A list containing the mean models for RV, FWHM, and S-index data.

        Raises
        ------
        KeyError
            If required keys are missing in the parameter vector.
        Exception
            If there is an error in computing the mean model.
        """
        try:
            # Initialize mean model arrays for RV, FWHM, and S-index
            Y0 = np.zeros(T0.shape)
            Y1 = np.zeros(self.adjusted_time_FWHM.shape) if self.include_fwhm else None
            Y2 = np.zeros(T0.shape) if self.include_sindex else None

            # Apply offsets to the RV data
            # ESPRESSO pre offset
            Y0[0 : self.n_pre] += q[self.Q["v0_pre"]]
            # ESPRESSO post offset
            Y0[self.n_pre : self.n_pre + self.n_post] += (
                q[self.Q["v0_pre"]] + q[self.Q["off_post"]]
            )
            # HARPS offset
            Y0[self.n_pre + self.n_post :] += (
                q[self.Q["v0_pre"]] + q[self.Q["off_harps"]]
            )

            if self.include_fwhm:
                # Apply offsets to the FWHM data
                # ESPRESSO pre offset
                Y1[0 : self.n_pre] += q[self.Q["C_FWHM_pre"]]
                # ESPRESSO post offset
                Y1[self.n_pre : self.n_pre + self.n_post] += q[self.Q["C_FWHM_post"]]

            if self.include_sindex:
                # Apply offsets to the S-index data
                # ESPRESSO pre offset
                Y2[0 : self.n_pre] += q[self.Q["C_Sindex_pre"]]
                # ESPRESSO post offset
                Y2[self.n_pre : self.n_pre + self.n_post] += q[self.Q["C_Sindex_post"]]
                # HARPS offset
                Y2[self.n_pre + self.n_post :] += q[self.Q["C_Sindex_harps"]]

            # Compute the RV model with corrected offsets
            Y0 += self.compute_planets_RV(T0, q)

            # Prepare the results
            results = [Y0, None, None]  # RV results are always included
            if self.include_fwhm:
                results[1] = Y1
            if self.include_sindex:
                results[2] = Y2

            return results

        except KeyError as e:
            print(f"Missing key in parameter vector: {e}")
            raise
        except Exception as e:
            print(f"Error in computing mean model: {e}")
            raise

    def myloglike(self, theta):
        """
        Computes the log likelihood for the given parameters using Gaussian Processes.

        This method calculates the log likelihood of the observed data given the
        parameter vector `theta`, using Gaussian Processes (GP) for modeling the
        Radial Velocity (RV), Full Width at Half Maximum (FWHM), and S-index data.

        Parameters
        ----------
        theta : array_like
            Input parameter vector.

        Returns
        -------
        log_likelihood : float
            The log likelihood of the observed data given the parameters.
        dps : list
            Derived parameters such as eccentricities.

        Raises
        ------
        KeyError
            If required keys are missing in the parameter vector.
        Exception
            If there is an error in computing the log likelihood.
        """
        try:
            q = np.copy(theta)

            # Derive parameters from the input vector
            self.derive_params(q)
            dps = (
                [self.e_b, self.e_c, self.e_d]
                if self.include_planet_b
                else [self.e_c, self.e_d]
            )

            # Log-transform certain parameters for GP
            A_RV = np.log(q[self.Q["A_RV"]])
            gamma = q[self.Q["gamma"]]
            log_period = np.log(q[self.Q["P_rot"]])
            t_decay = np.log(q[self.Q["t_decay"]])

            # Set GP parameter vector
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

            # Compute the GP model for RV data
            self.gp.compute(self.adjusted_time_RV, err_RV)

            # Compute the RV model and residuals
            mu_RV, mu_FWHM, mu_Sindex = self.mean_fxn(self.adjusted_time_RV, q)
            residuals_RV = self.obs_RV - mu_RV

            # Compute the log likelihood for RV data
            log_likelihood = self.gp.log_likelihood(residuals_RV)

            if self.include_fwhm:
                # GP parameters for FWHM
                A_FWHM = np.log(q[self.Q["A_FWHM"]])
                p1 = np.array([A_FWHM, gamma, log_period, t_decay])
                self.gp_fwhm.set_parameter_vector(p1)

                # Compute the FWHM error
                err_FWHM = np.block(
                    [
                        np.sqrt(
                            self.fwhm_err_pre**2 + q[self.Q["sigma_FWHM_pre"]] ** 2
                        ),
                        np.sqrt(
                            self.fwhm_err_post**2 + q[self.Q["sigma_FWHM_post"]] ** 2
                        ),
                    ]
                )

                # Compute the GP model for FWHM data
                self.gp_fwhm.compute(self.adjusted_time_FWHM, err_FWHM)

                # Compute the FWHM model and residuals
                residuals_FWHM = self.obs_FWHM - mu_FWHM
                log_likelihood += self.gp_fwhm.log_likelihood(residuals_FWHM)

            if self.include_sindex:
                # GP parameters for S-index
                A_Sindex = np.log(q[self.Q["A_Sindex"]])
                p2 = np.array([A_Sindex, gamma, log_period, t_decay])
                self.gp_sindex.set_parameter_vector(p2)

                # Compute the S-index error
                err_Sindex = np.block(
                    [
                        np.sqrt(
                            self.sindex_err_pre**2 + q[self.Q["sigma_Sindex_pre"]] ** 2
                        ),
                        np.sqrt(
                            self.sindex_err_post**2
                            + q[self.Q["sigma_Sindex_post"]] ** 2
                        ),
                        np.sqrt(
                            self.sindex_err_harps**2
                            + q[self.Q["sigma_Sindex_harps"]] ** 2
                        ),
                    ]
                )

                # Compute the GP model for S-index data
                self.gp_sindex.compute(self.adjusted_time_RV, err_Sindex)

                # Compute the S-index model and residuals
                residuals_Sindex = self.obs_Sindex - mu_Sindex
                log_likelihood += self.gp_sindex.log_likelihood(residuals_Sindex)

            return log_likelihood, dps

        except KeyError as e:
            print(f"Missing key in parameter vector: {e}")
            raise
        except Exception as e:
            print(f"Error in computing log likelihood: {e}")
            raise

    def dumper(self, live, dead, logweights, logZ, logZerr):
        """
        Callback function to process the state of the PolyChord run.

        This method is called during the PolyChord run to process and print information
        about the last dead point.

        Parameters
        ----------
        live : list
            List of live points.
        dead : list
            List of dead points.
        logweights : list
            Logarithm of the weights of the dead points.
        logZ : float
            Logarithm of the evidence.
        logZerr : float
            Error in the logarithm of the evidence.

        Raises
        ------
        IndexError
            If the dead list is empty.
        """
        try:
            print("Last dead point:", dead[-1])
        except IndexError as e:
            print(f"Error: {e}. The dead list is empty.")
            raise

    def run_analysis(self):
        """
        Runs the PolyChord analysis.

        This method sets up the PolyChord settings and runs the PolyChord analysis
        using the defined log-likelihood, prior, and dumper functions. It also records
        the runtime of the analysis.

        Raises
        ------
        Exception
            If there is an error during the PolyChord run.
        """
        try:
            # Setup PolyChord settings
            settings = PolyChordSettings(
                self.nDims,
                nDerived=self.nDerived,
                **self.polychord_settings,
                **self.output_params,
            )

            # Run PolyChord analysis
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

            # Record the runtime
            self.runtime = t_end - t_start

        except Exception as e:
            print(f"Error during PolyChord run: {e}")
            raise

    def load_samples_from_file(self, filename):
        """
        Loads samples from a specified file.

        This method reads the sample data from a text file in the output directory
        and returns it as a NumPy array.

        Parameters
        ----------
        filename : str
            Name of the file containing the sample data.

        Returns
        -------
        samples : np.ndarray
            The sample data loaded from the file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        IOError
            If there is an error in reading the file.
        """
        try:
            file_path = os.path.join(self.output_directory, filename)
            samples = np.loadtxt(file_path)
            return samples
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise
        except IOError as e:
            print(f"Error reading file: {e}")
            raise

    def handle_results(
        self, file_name="test_equal_weights.txt", plot_name="triangle_plot.pdf"
    ):
        """
        Handles the results of the analysis by loading samples, generating plots,
        and saving the output.

        This method loads the sample data from a specified file, generates a triangle
        plot of the posterior distributions, prints summary statistics, and saves
        the results to a pickle file.

        Parameters
        ----------
        file_name : str, optional
            Name of the file containing the sample data (default is "test_equal_weights.txt").
        plot_name : str, optional
            Name of the output plot file (default is "triangle_plot.pdf").

        Raises
        ------
        FileNotFoundError
            If the specified sample file does not exist.
        IOError
            If there is an error in reading the sample file or writing the pickle file.
        Exception
            If there is an error in processing the results or generating the plot.
        """
        try:
            # Generate a plot name based on included components
            plot_name = "triangle_planet_b_{}_fwhm_{}_sindex_{}.pdf".format(
                self.include_planet_b, self.include_fwhm, self.include_sindex
            )

            # Load the sample data from the specified file
            samples_file = self.load_samples_from_file(file_name)
            samples_data = samples_file

            # Define parameter names
            first_two_columns = ["log_likelihood", "Prior Volume"]
            param_names = first_two_columns + self.parameters_latex

            # Create an MCSamples object for posterior analysis
            posterior = MCSamples(samples=samples_data, names=param_names)
            posterior.setParamNames(param_names[1:])

            # Print summary statistics
            print(posterior.getMargeStats())

            # Compute means and standard deviations
            means = posterior.getMeans()
            vars = posterior.getVars()
            sds = np.sqrt(vars)

            # Calculate the total number of data points
            len_time = len(self.adjusted_time_RV)
            if self.include_fwhm:
                len_time += len(self.adjusted_time_FWHM)
            if self.include_sindex:
                len_time += len(self.adjusted_time_RV)

            # Print the number of dimensions and data points
            print("Number of dimensions: ", self.nDims)
            print("Number of data points:", len_time)

            # Print parameter names with their means and standard deviations
            for index, (name, mean, sd) in enumerate(zip(param_names[1:], means, sds)):
                print(rf"{index}. {name}: {mean} $\pm$ {sd}")

            # Generate and save the triangle plot
            g = gdplots.get_subplot_plotter()
            g.triangle_plot(posterior, filled=True)
            plot_path = os.path.join(self.output_directory, plot_name)
            g.export(plot_path)

            # Prepare the output data for pickling
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

            # Generate the pickle file name and save the results if runtime is long enough
            pickle_file_name = "pickle_planet_b_{}_fwhm_{}_sindex_{}.pickle".format(
                self.include_planet_b, self.include_fwhm, self.include_sindex
            )
            pickle_file_name = os.path.join(self.output_directory, pickle_file_name)
            print(f"L98-59 model with planet b {self.include_planet_b}, FWHM {self.include_fwhm}, S-index {self.include_sindex}")
            print("Finshed running in: {self.runtime}")
            print(f"Saving results to {self.output_directory, pickle_file_name}")
            print(f"Saving plots to {self.output_directory, plot_path}")

            if self.runtime > 25:
                with open(pickle_file_name, "wb") as f:
                    pickle.dump(algo_output, f)

        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise
        except IOError as e:
            print(f"Error reading file or writing pickle file: {e}")
            raise
        except Exception as e:
            print(f"Error processing results: {e}")
            raise
