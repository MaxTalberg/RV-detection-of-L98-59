from numpy import exp, log, sqrt
from scipy.special import erfcinv
from scipy.stats import beta, halfnorm


"""
Prior transform functions: map unit hyper-cube to actual parameter values,
in accordance with format required by PyMultiNest.
"""


def half_gaussian(theta, sigma):
    """Half-Gaussian prior with std dev. sigma"""
    theta_scaled = halfnorm.ppf(theta, scale=sigma)
    return theta_scaled


def gaussian(theta, mu, sigma):
    """Gaussian prior with mean mu and std dev. sigma"""
    theta_scaled = mu + sigma * (2.0**0.5) * erfcinv(2.0 * (1.0 - theta))
    return theta_scaled


def jeffreys(theta, a, b):
    """Log-uniform prior over [a,b]"""
    theta_scaled = exp(theta * (log(b) - log(a)) + log(a))
    return theta_scaled


def modjeffreys(theta, a, b):
    """Log-uniform prior over [0,b], with 'knee' at a"""
    theta_scaled = exp(theta * (log(1 + b / a)) + log(a)) - a
    return theta_scaled


def kipping_beta(theta):
    a = 0.867
    b = 3.03
    theta_scaled = beta.ppf(theta, a, b)
    return theta_scaled


def rayleigh(theta, sigma):
    """Truncated Rayleigh prior with scale parameter sigma"""
    theta_scaled = sigma * sqrt(-2 * log(1 - theta))
    return theta_scaled


def uniform(theta, a, b):
    """Uniform prior over [a,b]"""
    theta_scaled = a + (b - a) * theta
    return theta_scaled
