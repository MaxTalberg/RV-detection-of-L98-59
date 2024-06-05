import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config_data
import config_params
from computations import compute_derived_parameters


class TestComputations(unittest.TestCase):

    def test_derived_parameters(self):
        if config_params.INCLUDE_PLANET_B:
            q_input = [ 1.19990209e+00,  3.31729016e+02,  2.34520640e+03,  3.32474655e+00,
                        3.59450741e+00,  2.85999458e+00,  2.29255452e-02, -5.57909870e+03,
                        2.32550836e+00, -1.09038275e+02,  2.61008230e+00, 8.02978668e-01,
                        4.81020522e-02,  5.62639101e+00, -3.09879184e-02,  9.96513581e-01,
                        1.60791192e-01,  2.60259706e-01,  1.91950657e-01,  2.72160542e-01,
                        -1.89180781e+00,  7.96628954e-01,  3.86564906e+00,  3.30546041e-01,
                        1.29209696e-01,  1.64692577e+01, -7.89507989e-01,  7.92206473e-01
                        ]
            q_answer = [0.8044181429386744, 1928.758421910211, 0.3233885732210371, 1824.4738730676174, 0.35490256516568897, 1822.1665598801458]

            self.assertListEqual(compute_derived_parameters(q_input), q_answer)
        else:
            q_input = [ 8e+00,  2e+01,  5e+01,  2e+00,
                   1e+00,  5e-01,  9e-01, -5.579e+03,
                   1e+01, -1e+02,  2e+01,  2.0e-01,
                   3.1e-01,  1.3e+01, -7e-02, 1e-01,
                   2e+01,  1e-01,  2e-02,  8e+00,
                   -2e-01,  9.5e-01
                   ]
            q_answer = [0.36891733491393436, 1461.0260318357964, 0.10198039027185571, 1904.491183941788]

            self.assertListEqual(compute_derived_parameters(q_input), q_answer)
        
    def test_compute_planets_RV(self):
        pass

    def test_compute_offset(self):
        pass
    
if __name__ == '__main__':
    unittest.main()