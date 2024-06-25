import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from l9859_model import L9859Analysis
from pickle_data import unpickle_data


class TestL9859Analysis(unittest.TestCase):

    def setUp(self):
        self.filepath = "datasets/cleaned_data_20240531.pickle"
        self.include_planet_b = True
        self.include_fwhm = True
        self.include_sindex = True
        self.polychord_settings = None
        self.algorithm_params = None
