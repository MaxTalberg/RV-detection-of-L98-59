import os
import sys


from config_params import INCLUDE_PLANET_B
from computations import compute_derived_parameters

INCLUDE_PLANET_B = True

q_input = [
    1.19990209e00,
    3.31729016e02,
    2.34520640e03,
    3.32474655e00,
    3.59450741e00,
    2.85999458e00,
    2.29255452e-02,
    -5.57909870e03,
    2.32550836e00,
    -1.09038275e02,
    2.61008230e00,
    8.02978668e-01,
    4.81020522e-02,
    5.62639101e00,
    -3.09879184e-02,
    9.96513581e-01,
    1.60791192e-01,
    2.60259706e-01,
    1.91950657e-01,
    2.72160542e-01,
    -1.89180781e00,
    7.96628954e-01,
    3.86564906e00,
    3.30546041e-01,
    1.29209696e-01,
    1.64692577e01,
    -7.89507989e-01,
    7.92206473e-01,
]


print(compute_derived_parameters(q_input))
