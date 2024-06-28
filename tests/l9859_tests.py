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

    @patch("pickle_data.unpickle_data", side_effect=IOError("File not found"))
    def test_load_data_failure(self, mock_unpickle_data):
        with self.assertRaises(IOError):
            unpickle_data("non_existent_file.pickle")

    def test_initialise_parameters_default(self):
        self.analysis.initialise_parameters()
        # Test the default setup without additional flags
        self.assertIn("A_RV", self.analysis.parameters)
        self.assertIn("A_FWHM", self.analysis.parameters)
        self.assertIn("A_Sindex", self.analysis.parameters)
        self.assertEqual(len(self.analysis.parameters), 40)
        self.assertEqual(self.analysis.nDims, len(self.analysis.parameters))
        self.assertEqual(self.analysis.nDerived, 3)


# Run the tests
if __name__ == "__main__":
    unittest.main()
