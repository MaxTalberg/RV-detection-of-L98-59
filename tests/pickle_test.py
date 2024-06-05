import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
from pickle_data import load_data_from_pickle


class TestPickles(unittest.TestCase):
    def test_data_loading(self):
        X = load_data_from_pickle("datasets/cleaned_data_20240531.pickle")
        assert (
            "ESPRESSO_pre" in X and "ESPRESSO_post" in X and "HARPS" in X
        ), "Dataset keys missing"
        assert len(X["ESPRESSO_pre"]["RV"]) > 0, "Data loading issue for ESPRESSO_pre"
        assert len(X["ESPRESSO_post"]["RV"]) > 0, "Data loading issue for ESPRESSO_post"
        assert len(X["HARPS"]["RV"]) > 0, "Data loading issue for HARPS"


if __name__ == "__main__":
    unittest.main()
