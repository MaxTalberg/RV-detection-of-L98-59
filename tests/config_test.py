import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config_data


class TestConfig(unittest.TestCase):

    def test_data(self):
        # Assertions to verify all data is loaded correctly
        self.assertEqual(config_data.time_pre[0], 2458436.80567403)
        self.assertEqual(config_data.adjusted_time_RV[0], config_data.time_pre[0] - 2457000)
        self.assertEqual(config_data.max_jitter_harps, 10.5)
        self.assertEqual(config_data.n_pre, 39)

    def test_params(self):
        # Assertions to verify all parameters are loaded correctly
        self.assertEqual(config_data.Q['A_RV'], 0)
        self.assertEqual(config_data.Q['P_rot'], 1)
        self.assertEqual(config_data.Q['t_decay'], 2)
        self.assertEqual(config_data.Q['gamma'], 3)
        if config_data.INCLUDE_PLANET_B:
            self.assertEqual(len(config_data.Q), 34)
            self.assertEqual(config_data.nDims, 28)
            self.assertEqual(config_data.nDerived, 6)
        else:
            self.assertEqual(len(config_data.Q), 26)
            self.assertEqual(config_data.nDims, 22)
            self.assertEqual(config_data.nDerived, 4)

if __name__ == '__main__':
    unittest.main()