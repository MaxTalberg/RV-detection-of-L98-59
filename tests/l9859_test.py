import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from l9859_model import L9859Analysis
from pickle_data import unpickle_data


class TestL9859(unittest.TestCase):
    """
    Unit tests for the L98-59 model class, ensuring that initialization and settings
    are correctly applied and verifying the behavior of its setup.
    """

    @patch("l9859_model.L9859Analysis.load_data")
    def setUp(self, mock_load_data):
        """
        Set up method for test cases. Initialises the L9859Analysis with specific
        parameters before each test.

        Parameters
        ----------
        mock_load_data : unittest.mock.patch
            Patch for the `load_data` method to prevent actual data loading during initialization.
        """
        # Initialise the L9859Analysis object with predefined parameters and mocks
        self.analysis = L9859Analysis(
            filepath="../datasets/cleaned_data_20240531.pickle",
            include_planet_b=True,
            include_fwhm=True,
            include_sindex=True,
        )

    def test_init(self):
        """
        Test the initialization of the L9859Analysis class to ensure all properties
        are set as expected.
        """
        # Check the filepath is stored correctly
        self.assertEqual(
            self.analysis.filepath, "../datasets/cleaned_data_20240531.pickle"
        )

        # Check feature inclusion flags
        self.assertTrue(self.analysis.include_planet_b, "Planet B should be included.")
        self.assertTrue(self.analysis.include_fwhm, "FWHM should be included.")
        self.assertTrue(self.analysis.include_sindex, "S-index should be included.")

        # Check types and values of other properties
        self.assertIsInstance(
            self.analysis.polychord_settings,
            dict,
            "Polychord settings should be a dictionary.",
        )
        self.assertIn(
            "do_clustering",
            self.analysis.polychord_settings,
            "Clustering setting should be in polychord settings.",
        )

        # Verify output directory parameter
        self.assertEqual(
            self.analysis.output_params["base_dir"],
            "output_dir/specific_run",
            "Output base directory should match specified configuration.",
        )

    @patch("pickle_data.unpickle_data")
    def test_load_data(self, mock_unpickle_data):
        """
        Test successful data loading to ensure data structures are correctly initialised
        and populated with data from the unpickle process.

        Parameters
        ----------
        mock_unpickle_data : unittest.mock.patch
            Mock of the unpickle_data function, set to return a predefined dataset.
        """
        # Set up mock return values for unpickle_data to simulate loaded data
        mock_unpickle_data.return_value = {
            "ESPRESSO_pre": {
                "RV": [],
                "Time": [],
                "FWHM": [],
                "e_FWHM": [],
                "Sindex": [],
                "e_Sindex": [],
            },
            "ESPRESSO_post": {
                "RV": [],
                "Time": [],
                "FWHM": [],
                "e_FWHM": [],
                "Sindex": [],
                "e_Sindex": [],
            },
            "HARPS": {"RV": [], "Time": []},
        }

        # Invoke the method to load data
        self.analysis.load_data()

        # Assertions to ensure that data containers are not None after loading
        self.assertIsNotNone(
            self.analysis.X_pre, "X_pre should not be None after loading data."
        )
        self.assertIsNotNone(
            self.analysis.X_post, "X_post should not be None after loading data."
        )
        self.assertIsNotNone(
            self.analysis.X_harps, "X_harps should not be None after loading data."
        )

        # Ensure data setup is called correctly
        self.analysis.setup_data()

    @patch("pickle_data.unpickle_data", side_effect=IOError("File not found"))
    def test_load_data_failure(self, mock_unpickle_data):
        """
        Test the error handling of the data loading process when the data file is missing,
        expecting an IOError.

        Parameters
        ----------
        mock_unpickle_data : unittest.mock.patch
            Mock of the unpickle_data function, set to raise an IOError.
        """
        # Expect an IOError when attempting to load a non-existent file
        with self.assertRaises(
            IOError, msg="Should raise IOError if the data file is missing"
        ):
            unpickle_data("non_existent_file.pickle")

    def test_initialise_parameters_default(self):
        """
        Test the initialization of parameters with all additional features (planet B, FWHM, S-index) turned off.
        This test verifies the base parameter setup without extensions.
        """
        # Reinitialise analysis with no additional features
        self.analysis = L9859Analysis(
            filepath="../datasets/cleaned_data_20240531.pickle",
            include_planet_b=False,
            include_fwhm=False,
            include_sindex=False,
        )
        self.analysis.initialise_parameters()

        # Verify only base parameters are included
        self.assertIn("A_RV", self.analysis.parameters, "A_RV should be in parameters")
        self.assertNotIn(
            "A_FWHM", self.analysis.parameters, "A_FWHM should not be in parameters"
        )
        self.assertNotIn(
            "A_Sindex", self.analysis.parameters, "A_Sindex should not be in parameters"
        )

        # Check the total number of parameters and derived parameters
        self.assertEqual(
            len(self.analysis.parameters),
            22,
            "Total parameters should match the expected count without features",
        )
        self.assertEqual(
            self.analysis.nDims,
            len(self.analysis.parameters),
            "nDims should match the number of parameters",
        )
        self.assertEqual(
            self.analysis.nDerived,
            2,
            "nDerived should match expected count for base features only",
        )

    def test_initialise_parameters_full(self):
        """
        Test the initialization of parameters with all additional features (planet B, FWHM, S-index) turned on.
        This test verifies the complete parameter setup including extensions.
        """
        # Assume analysis is initialised in the setUp with all features enabled
        self.analysis.initialise_parameters()

        # Verify all parameters, including those for additional features, are included
        self.assertIn("A_RV", self.analysis.parameters, "A_RV should be in parameters")
        self.assertIn(
            "A_FWHM", self.analysis.parameters, "A_FWHM should be in parameters"
        )
        self.assertIn(
            "A_Sindex", self.analysis.parameters, "A_Sindex should be in parameters"
        )

        # Check the total number of parameters and derived parameters
        self.assertEqual(
            len(self.analysis.parameters),
            40,
            "Total parameters should match the expected count with all features",
        )
        self.assertEqual(
            self.analysis.nDims,
            len(self.analysis.parameters),
            "nDims should match the number of parameters",
        )
        self.assertEqual(
            self.analysis.nDerived,
            3,
            "nDerived should match expected count with all features enabled",
        )

    def test_derive_params(self):
        """
        Test the derivation of eccentricity parameters based on the given values of secosw and sesinw for planets b, c, and d.
        """
        # Set up mock values for Q and q to test derivation logic
        self.analysis.Q = {
            "secosw_b": 0,
            "sesinw_b": 1,
            "secosw_c": 2,
            "sesinw_c": 3,
            "secosw_d": 4,
            "sesinw_d": 5,
        }
        q_values = [
            0.6,
            0.8,
            0.3,
            0.4,
            0.5,
            0.5,
        ]  # Mock values for secosw, sesinw for planets b, c, d

        # Invoke derive_params to calculate eccentricities based on q_values
        self.analysis.derive_params(q_values)

        # Calculate expected eccentricities manually for verification
        expected_e_b = np.sqrt(q_values[0] ** 2 + q_values[1] ** 2)
        expected_e_c = np.sqrt(q_values[2] ** 2 + q_values[3] ** 2)
        expected_e_d = np.sqrt(q_values[4] ** 2 + q_values[5] ** 2)

        # Verify the derived parameters match expected values
        if self.analysis.include_planet_b:
            self.assertAlmostEqual(
                self.analysis.e_b,
                expected_e_b,
                "Eccentricity of planet b does not match expected value",
            )
        self.assertAlmostEqual(
            self.analysis.e_c,
            expected_e_c,
            "Eccentricity of planet c does not match expected value",
        )
        self.assertAlmostEqual(
            self.analysis.e_d,
            expected_e_d,
            "Eccentricity of planet d does not match expected value",
        )

    @patch("radvel.RVModel", autospec=True)
    @patch("radvel.Parameters", autospec=True)
    def test_compute_planets_RV(self, mock_params, mock_rv_model):
        """
        Test the computation of radial velocities (RV) using the radvel model with mocked parameters and RV model.
        """
        # Setup necessary data and parameters
        self.analysis.load_data()  # Assume load_data populates X_pre, X_post, and X_harps appropriately
        self.analysis.setup_data()
        self.analysis.initialise_parameters()

        # Set a time array for RV computations
        self.T0 = self.analysis.time_RV

        # Setup mock objects to simulate radvel behavior
        mock_params.return_value = MagicMock()
        mock_rv_model_instance = MagicMock()
        mock_rv_model.return_value = mock_rv_model_instance
        mock_rv_model_instance.return_value = np.random.normal(
            0, 1, len(self.T0)
        )  # Mock radial velocity data

        # Run RV computation with random parameter values
        q_vals = np.random.rand(len(self.analysis.parameters))
        rv_total = self.analysis.compute_planets_RV(self.T0, q_vals)

        # Assert RV data is as expected
        self.assertIsInstance(
            rv_total, np.ndarray, "Returned RV data should be a numpy array"
        )
        self.assertEqual(
            len(rv_total),
            len(self.T0),
            "Length of RV data should match length of input times",
        )
        mock_params.assert_called_once()
        mock_rv_model.assert_called_once_with(mock_params.return_value)

    @patch("prior_transforms.gaussian")
    @patch("prior_transforms.kipping_beta")
    @patch("prior_transforms.uniform")
    def test_planet_prior(self, mock_uniform, mock_kipping_beta, mock_gaussian):
        """
        Test the planet_prior function to ensure it correctly transforms parameters using the specified prior distributions.
        """
        # Configure mock behavior for each prior function to return predictable outputs
        mock_gaussian.side_effect = lambda x, mu, sigma: mu  # Returns mean
        mock_kipping_beta.side_effect = lambda x: 0.2  # Returns a fixed value
        mock_uniform.side_effect = (
            lambda x, low, high: (low + high) / 2
        )  # Returns midpoint

        # Initialise parameter values for testing
        q_vals = np.full(
            len(self.analysis.parameters), 0.5
        )  # Base value for simplicity
        transformed_qq = self.analysis.planet_prior(q_vals.copy())

        # Check that the transformed values are as expected
        if self.analysis.include_planet_b:
            self.assertAlmostEqual(
                transformed_qq[self.analysis.Q["P_b"]], 2.2531136, places=5
            )
        self.assertAlmostEqual(
            transformed_qq[self.analysis.Q["P_c"]], 3.6906777, places=5
        )
        self.assertAlmostEqual(
            transformed_qq[self.analysis.Q["secosw_c"]], 0.2, places=5
        )
        self.assertAlmostEqual(
            transformed_qq[self.analysis.Q["w_c"]], 0, places=5
        )  # Uniform distribution centered at 0

        # Confirm the correct number of calls to each mocked prior function
        self.assertEqual(mock_gaussian.call_count, 6)
        self.assertEqual(mock_kipping_beta.call_count, 6)
        self.assertEqual(mock_uniform.call_count, 6)

    @patch("prior_transforms.uniform")
    @patch("prior_transforms.jeffreys")
    @patch("prior_transforms.gaussian")
    def test_myprior(self, mock_gaussian, mock_jeffreys, mock_uniform):
        """
        Test the myprior function to ensure proper application of parameter priors across different parameter types.
        """
        # Setup mock returns that mimic the midpoint or mean behavior of the priors
        mock_uniform.side_effect = lambda x, low, high: (low + high) / 2
        mock_jeffreys.side_effect = lambda x, a, b: (a + b) / 2
        mock_gaussian.side_effect = lambda x, mu, sigma: mu

        # Create a parameter array to test
        q_vals = np.full(len(self.analysis.Q), 0.5)
        transformed_qq = self.analysis.myprior(q_vals.copy())

        # Expected values are determined by the midpoint or mean of the prior distributions
        expected_values = {
            self.analysis.Q["A_RV"]: 8.4,
            self.analysis.Q["P_rot"]: 262.5,
            self.analysis.Q["t_decay"]: (262.5 / 2 + 2600) / 2,
            self.analysis.Q["gamma"]: 2.525,
            self.analysis.Q["sigma_RV_pre"]: 1.985295,
            self.analysis.Q["sigma_RV_post"]: 1.64266,
            self.analysis.Q["sigma_RV_harps"]: 5.25,
            self.analysis.Q["v0_pre"]: -5579.2,
            self.analysis.Q["off_post"]: 2.86,
            self.analysis.Q["off_harps"]: -99.4,
        }

        # Assert the expected values match the transformed values
        for param, index in self.analysis.Q.items():
            if index in expected_values:
                self.assertAlmostEqual(
                    transformed_qq[index],
                    expected_values[index],
                    places=5,
                    msg=f"Parameter {param} at index {index} did not match expected.",
                )

        # Check the called status of mocks to ensure they were used as expected
        self.assertTrue(mock_uniform.called)
        self.assertTrue(mock_jeffreys.called)
        self.assertTrue(mock_gaussian.called)

    def test_mean_fxn(self):
        """
        Test the mean_fxn method to ensure it correctly computes mean functions for RV, FWHM, and Sindex based on the provided time and parameter vector.
        """
        # Prepare necessary data and parameter setup
        self.analysis.load_data()  # Load the mocked data
        self.analysis.setup_data()  # Set up data structures
        self.analysis.initialise_parameters()  # Initialise parameters for the model

        self.T0 = self.analysis.time_RV
        q = np.full(len(self.analysis.Q), 1.0)  # Placeholder values for parameters

        # Execute the mean function computation
        results = self.analysis.mean_fxn(self.T0, q)

        # Verify the output shapes and null conditions based on configuration flags
        self.assertEqual(results[0].shape, self.T0.shape, "Y0 results shape mismatch.")
        if self.analysis.include_fwhm:
            self.assertEqual(
                results[1].shape,
                self.analysis.adjusted_time_FWHM.shape,
                "Y1 results shape mismatch.",
            )
        else:
            self.assertIsNone(
                results[1], "Y1 should be None when FWHM is not included."
            )
        if self.analysis.include_sindex:
            self.assertEqual(
                results[2].shape, self.T0.shape, "Y2 results shape mismatch."
            )
        else:
            self.assertIsNone(
                results[2], "Y2 should be None when Sindex is not included."
            )

    def test_myloglike(self):
        """
        Test the myloglike method to evaluate the log likelihood calculation for the given parameter vector.
        """
        # Initialise parameter vector for the test
        theta = np.full(len(self.analysis.parameters), 1.0)
        self.analysis.load_data()  # Load mocked data
        self.analysis.setup_data()  # Configure data structures
        self.analysis.initialise_parameters()  # Set up model parameters

        # Compute log likelihood and derived parameters
        loglike, dps = self.analysis.myloglike(theta)

        # Assess the output, checking the number of derived parameters and the log likelihood value
        self.assertEqual(len(dps), 3, "Derived parameters count mismatch.")
        self.assertIsNotNone(loglike, "Log likelihood should be a non-None value.")


# Run the tests
if __name__ == "__main__":
    unittest.main()
