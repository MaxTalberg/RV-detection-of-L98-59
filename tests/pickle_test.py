import os
import sys
from unittest.mock import mock_open, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
from pickle_data import unpickle_data


class TestPickles(unittest.TestCase):
    """
    Unit tests for the unpickle_data function.
    """

    @patch("builtins.open", new_callable=mock_open, read_data=b"")
    def test_data_loading(self, mock_file):
        """
        Tests successful loading of data from the pickle file.
        """
        mock_data = {
            "ESPRESSO_pre": {"RV": [1, 2, 3]},
            "ESPRESSO_post": {"RV": [4, 5, 6]},
            "HARPS": {"RV": [7, 8, 9]},
        }
        with patch("pickle.load", return_value=mock_data):
            X = unpickle_data("../datasets/cleaned_data_20240531.pickle")
            self.assertIn("ESPRESSO_pre", X, "Dataset keys missing")
            self.assertIn("ESPRESSO_post", X, "Dataset keys missing")
            self.assertIn("HARPS", X, "Dataset keys missing")
            self.assertGreater(
                len(X["ESPRESSO_pre"]["RV"]), 0, "Data loading issue for ESPRESSO_pre"
            )
            self.assertGreater(
                len(X["ESPRESSO_post"]["RV"]), 0, "Data loading issue for ESPRESSO_post"
            )
            self.assertGreater(len(X["HARPS"]["RV"]), 0, "Data loading issue for HARPS")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_file_not_found(self, mock_file):
        """
        Tests that FileNotFoundError is raised when the file does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            unpickle_data("non_existent_file.pickle")

    @patch("builtins.open", new_callable=mock_open, read_data=b"")
    def test_eof_error(self, mock_file):
        """
        Tests that EOFError is raised for an empty or improperly formatted file.
        """
        with patch("pickle.load", side_effect=EOFError):
            with self.assertRaises(EOFError):
                unpickle_data("empty_or_corrupt_file.pickle")

    @patch("builtins.open", new_callable=mock_open, read_data=b"")
    def test_generic_exception(self, mock_file):
        """
        Tests that a generic exception is raised for other issues during loading.
        """
        with patch("pickle.load", side_effect=Exception("Generic error")):
            with self.assertRaises(Exception):
                unpickle_data("generic_error_file.pickle")


if __name__ == "__main__":
    unittest.main()
