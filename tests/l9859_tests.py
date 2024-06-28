import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from l9859_model import L9859Analysis
from pickle_data import unpickle_data


"""class TestL9859Analysis(unittest.TestCase):

    def setUp(self):
        self.filepath = "datasets/cleaned_data_20240531.pickle"
        self.include_planet_b = True
        self.include_fwhm = True
        self.include_sindex = True
        self.polychord_settings = None
        self.algorithm_params = None"""


class TestL9859Analysis(unittest.TestCase):

    @patch("l9859_model.L9859Analysis.load_data")
    def setUp(self, mock_load_data):
        # Mock the load_data, initialise_parameters, and create_qp_kernel to not perform any action during instantiation
        self.analysis = L9859Analysis(
            filepath="datasets/cleaned_data_20240531.pickle",
            include_planet_b=True,
            include_fwhm=True,
            include_sindex=True,
        )

    def test_init(self):
        self.assertEqual(
            self.analysis.filepath, "datasets/cleaned_data_20240531.pickle"
        )
        self.assertTrue(self.analysis.include_planet_b)
        self.assertTrue(self.analysis.include_fwhm)
        self.assertTrue(self.analysis.include_sindex)
        self.assertIsInstance(self.analysis.polychord_settings, dict)
        self.assertIn("do_clustering", self.analysis.polychord_settings)
        self.assertEqual(
            self.analysis.output_params["base_dir"], "output_dir/specific_run"
        )

    @patch("pickle_data.unpickle_data")
    def test_load_data(self, mock_unpickle_data):
        # Set up mock return values for unpickle_data
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
        self.analysis.load_data()  # Call the method
        self.assertIsNotNone(self.analysis.X_pre)
        self.assertIsNotNone(self.analysis.X_post)
        self.assertIsNotNone(self.analysis.X_harps)
        self.analysis.setup_data()

    @patch("pickle_data.unpickle_data", side_effect=IOError("File not found"))
    def test_load_data_failure(self, mock_unpickle_data):
        with self.assertRaises(IOError):
            unpickle_data("non_existent_file.pickle")

    def test_initialise_parameters_default(self):
        self.analysis = L9859Analysis(
            filepath="datasets/cleaned_data_20240531.pickle",
            include_planet_b=False,
            include_fwhm=False,
            include_sindex=False,
        )
        self.analysis.initialise_parameters()
        # Test the default setup without additional flags
        self.assertIn("A_RV", self.analysis.parameters)
        self.assertNotIn("A_FWHM", self.analysis.parameters)
        self.assertNotIn("A_Sindex", self.analysis.parameters)
        self.assertEqual(len(self.analysis.parameters), 22)
        self.assertEqual(self.analysis.nDims, len(self.analysis.parameters))
        self.assertEqual(self.analysis.nDerived, 2)

    def test_initialise_parameters_full(self):
        self.analysis.initialise_parameters()
        # Test the default setup without additional flags
        self.assertIn("A_RV", self.analysis.parameters)
        self.assertIn("A_FWHM", self.analysis.parameters)
        self.assertIn("A_Sindex", self.analysis.parameters)
        self.assertEqual(len(self.analysis.parameters), 40)
        self.assertEqual(self.analysis.nDims, len(self.analysis.parameters))
        self.assertEqual(self.analysis.nDerived, 3)

    def test_derive_params(self):
        # Set up initial values for Q and q for testing
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
        ]  # Mock values for secosw, sesinw for b, c, d

        # Invoke derive_params with the test values
        self.analysis.derive_params(q_values)

        # Expected values calculated manually for confirmation
        expected_e_b = np.sqrt(
            q_values[self.analysis.Q["secosw_b"]] ** 2
            + q_values[self.analysis.Q["sesinw_b"]] ** 2
        )
        expected_e_c = np.sqrt(
            q_values[self.analysis.Q["secosw_c"]] ** 2
            + q_values[self.analysis.Q["sesinw_c"]] ** 2
        )
        expected_e_d = np.sqrt(
            q_values[self.analysis.Q["secosw_d"]] ** 2
            + q_values[self.analysis.Q["sesinw_d"]] ** 2
        )

        # Assert the correct calculations of derived parameters
        if self.analysis.include_planet_b:
            self.assertAlmostEqual(self.analysis.e_b, expected_e_b)
        self.assertAlmostEqual(self.analysis.e_c, expected_e_c)
        self.assertAlmostEqual(self.analysis.e_d, expected_e_d)

    @patch("radvel.RVModel", autospec=True)
    @patch("radvel.Parameters", autospec=True)
    def test_compute_planets_RV(self, mock_params, mock_rv_model):
        self.analysis.load_data()  # Load the mocked data
        self.analysis.setup_data()  # Set up the data based on loaded data
        self.analysis.initialise_parameters()  # Set up the parameters

        self.T0 = self.analysis.time_RV
        # Setup mock return behavior
        mock_params.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_rv_model.return_value = mock_model_instance
        mock_model_instance.return_value = np.random.normal(
            0, 1, len(self.T0)
        )  # Mock RV data

        # Compute RV
        q_vals = np.random.rand(len(self.analysis.parameters))
        rv_total = self.analysis.compute_planets_RV(self.T0, q_vals)

        # Assertions to verify behavior
        self.assertIsInstance(rv_total, np.ndarray)
        self.assertEqual(len(rv_total), len(self.T0))
        mock_params.assert_called()
        mock_rv_model.assert_called_with(mock_params.return_value)

    @patch("prior_transforms.gaussian")
    @patch("prior_transforms.kipping_beta")
    @patch("prior_transforms.uniform")
    def test_planet_prior(self, mock_uniform, mock_kipping_beta, mock_gaussian):
        # Set the mock functions to return specific, controlled values directly
        mock_gaussian.side_effect = (
            lambda x, mu, sigma: mu
        )  # Ignore x and sigma, just return mu
        mock_kipping_beta.side_effect = (
            lambda x: 0.2
        )  # Constant value, simplifying the test
        mock_uniform.side_effect = (
            lambda x, low, high: (low + high) / 2
        )  # Midpoint of low and high

        # Set an initial value where the expected outcome is known
        q_vals = np.full(
            len(self.analysis.parameters), 0.5
        )  # Using 0.5 as a simple base for calculation
        transformed_qq = self.analysis.planet_prior(q_vals.copy())

        # Using assertAlmostEqual for floating-point comparison with a tolerance
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
        )  # As uniform from -π to π centers at 0

        # Verify calls to the mocked functions
        self.assertEqual(mock_gaussian.call_count, 6)  # 3 planets * (P and Tc)
        self.assertEqual(
            mock_kipping_beta.call_count, 6
        )  # 3 planets * (secosw and sesinw)
        self.assertEqual(mock_uniform.call_count, 6)

    @patch("prior_transforms.uniform")
    @patch("prior_transforms.jeffreys")
    @patch("prior_transforms.gaussian")
    def test_myprior(self, mock_gaussian, mock_jeffreys, mock_uniform):
        # Set the mock functions to return specific, controlled values directly
        mock_uniform.side_effect = lambda x, low, high: (low + high) / 2
        mock_jeffreys.side_effect = lambda x, a, b: (a + b) / 2
        mock_gaussian.side_effect = lambda x, mu, sigma: mu

        # Initialize the analysis instance
        self.analysis = L9859Analysis(
            filepath="datasets/cleaned_data_20240531.pickle",
            include_planet_b=True,  # Adjust as needed
            include_fwhm=True,  # Adjust as needed
            include_sindex=True,  # Adjust as needed
        )

        # Create an array of test input values
        q_vals = np.full(
            len(self.analysis.Q), 0.5
        )  # Using 0.5 as a simple base for calculation
        transformed_qq = self.analysis.myprior(q_vals.copy())

        # Define expected values based on mock behavior
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

        # Assert transformed values
        for param, index in self.analysis.Q.items():
            if index in expected_values:
                self.assertAlmostEqual(
                    transformed_qq[index],
                    expected_values[index],
                    places=5,
                    msg=f"Failed at parameter {param} (index {index})",
                )

        # Verify that each mock was called correctly
        self.assertTrue(mock_uniform.called)
        self.assertTrue(mock_jeffreys.called)
        self.assertTrue(mock_gaussian.called)

    def test_mean_fxn(self):
        # Create mock values for T0 and q
        self.analysis.load_data()  # Load the mocked data
        self.analysis.setup_data()  # Set up the data based on loaded data
        self.analysis.initialise_parameters()  # Set up the parameters

        self.T0 = self.analysis.time_RV

        q = np.full(
            len(self.analysis.Q), 1.0
        )  # Example q array with placeholder values

        # Call the mean_fxn method
        results = self.analysis.mean_fxn(self.T0, q)

        # Assert that results contain expected shapes
        self.assertEqual(
            results[0].shape, self.T0.shape
        )  # Y0 should have the same shape as T0
        if self.analysis.include_fwhm:
            self.assertEqual(results[1].shape, self.analysis.adjusted_time_FWHM.shape)
        else:
            self.assertIsNone(results[1])  # Y1 should be None if include_fwhm is False
        if self.analysis.include_sindex:
            self.assertEqual(
                results[2].shape, self.T0.shape
            )  # Y2 should have the same shape as T0
        else:
            self.assertIsNone(
                results[2]
            )  # Y2 should be None if include_sindex is False

    def test_myloglike(self):
        # Create mock values for theta (parameter vector)
        theta = np.full(len(self.analysis.parameters), 1.0)
        self.analysis.load_data()  # Load the mocked data
        self.analysis.setup_data()  # Set up the data based on loaded data
        self.analysis.initialise_parameters()

        # Call the myloglike method
        loglike, dps = self.analysis.myloglike(theta)

        # Assert that the results are as expected
        self.assertEqual(len(dps), 3)
        self.assertIsNotNone(loglike)


# Run the tests
if __name__ == "__main__":
    unittest.main()
