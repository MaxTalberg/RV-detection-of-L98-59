import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
from unittest.mock import patch, MagicMock

import prior_transforms as pt
from run_L9859 import planet_prior

# Assuming the `planet_prior`, `Q`, and `pt` are imported or defined in the scope where this test will run.

class TestPlanetPrior(unittest.TestCase):

    def setUp(self):
        # Setup a sample Q dictionary
        self.Q = {
            'P_b': 0, 'secosw_b': 1, 'sesinw_b': 2, 'K_b': 3, 'w*_b': 4, 'phi_b': 5,
            'P_c': 6, 'secosw_c': 7, 'sesinw_c': 8, 'K_c': 9, 'w*_c': 10, 'phi_c': 11,
            'P_d': 12, 'secosw_d': 13, 'sesinw_d': 14, 'K_d': 15, 'w*_d': 16, 'phi_d': 17
        }

        # Mock the priors to simply return the value plus 100 for differentiation in tests
        pt.jeffreys = MagicMock(side_effect=lambda x, low, high: x + 100)
        pt.kipping_beta = MagicMock(side_effect=lambda x: x + 100)
        pt.uniform = MagicMock(side_effect=lambda x, low, high: x + 100)

    def test_planet_prior_includes_planet_b(self):
        qq = {key: index for index, key in enumerate(self.Q)}
        result = planet_prior(qq, INCLUDE_PLANET_B=True)
        
        # Check that planet B's parameters were transformed
        self.assertEqual(result[self.Q['P_b']], 100)
        self.assertEqual(result[self.Q['secosw_b']], 101)
        self.assertEqual(result[self.Q['sesinw_b']], 102)
        self.assertEqual(result[self.Q['K_b']], 103)
        self.assertEqual(result[self.Q['w*_b']], 104)
        self.assertEqual(result[self.Q['phi_b']], 105)

        # Check that planet C's and D's parameters were also transformed
        self.assertEqual(result[self.Q['P_c']], 106 + 100)
        self.assertEqual(result[self.Q['K_c']], 109 + 100)

    def test_planet_prior_excludes_planet_b(self):
        qq = {key: index for index, key in enumerate(self.Q)}
        result = planet_prior(qq, INCLUDE_PLANET_B=False)
        
        # Check that planet B's parameters were not transformed
        self.assertEqual(result[self.Q['P_b']], 0)
        self.assertEqual(result[self.Q['secosw_b']], 1)

        # Check that planet C's and D's parameters were transformed
        self.assertEqual(result[self.Q['P_c']], 106 + 100)
        self.assertEqual(result[self.Q['K_c']], 109 + 100)

if __name__ == '__main__':
    unittest.main()
