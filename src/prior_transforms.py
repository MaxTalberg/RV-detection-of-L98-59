from numpy import exp, log, sqrt
from scipy.special import erfcinv
from scipy.stats import beta, halfnorm


"""
Prior transform functions: map unit hyper-cube to actual parameter values,
in accordance with format required by PyMultiNest.
"""


def half_gaussian(theta, sigma):
    """
    Transforms a uniform variable into a half-Gaussian distributed variable.

    Parameters
    ----------
    theta : float
        Uniformly distributed variable [0, 1] from the unit hypercube.
    sigma : float
        Standard deviation of the desired half-Gaussian distribution.

    Returns
    -------
    float
        A value sampled from a half-Gaussian distribution.
    """
    theta_scaled = halfnorm.ppf(theta, scale=sigma)
    return theta_scaled


def gaussian(theta, mu, sigma):
    """
    Transforms a uniform variable into a Gaussian distributed variable.

    Parameters
    ----------
    theta : float
        Uniformly distributed variable [0, 1] from the unit hypercube.
    mu : float
        Mean of the desired Gaussian distribution.
    sigma : float
        Standard deviation of the desired Gaussian distribution.

    Returns
    -------
    float
        A value sampled from a Gaussian distribution.
    """
    theta_scaled = mu + sigma * (2.0**0.5) * erfcinv(2.0 * (1.0 - theta))
    return theta_scaled


def jeffreys(theta, a, b):
    """
    Transforms a uniform variable into a value sampled from a Jeffreys prior (log-uniform distribution).

    Parameters
    ----------
    theta : float
        Uniformly distributed variable [0, 1] from the unit hypercube.
    a : float
        Lower bound of the distribution.
    b : float
        Upper bound of the distribution.

    Returns
    -------
    float
        A value sampled from a log-uniform distribution spanning [a, b].
    """
    theta_scaled = exp(theta * (log(b) - log(a)) + log(a))
    return theta_scaled


def modjeffreys(theta, a, b):
    """
    Transforms a uniform variable into a value sampled from a modified Jeffreys prior.

    Parameters
    ----------
    theta : float
        Uniformly distributed variable [0, 1] from the unit hypercube.
    a : float
        Knee location of the distribution.
    b : float
        Upper bound of the distribution.

    Returns
    -------
    float
        A value sampled from a modified log-uniform distribution.
    """
    theta_scaled = exp(theta * (log(1 + b / a)) + log(a)) - a
    return theta_scaled


def kipping_beta(theta):
    """
    Transforms a uniform variable into a value sampled from a Kipping Beta distribution.

    Parameters
    ----------
    theta : float
        Uniformly distributed variable [0, 1] from the unit hypercube.

    Returns
    -------
    float
        A value sampled from a Beta distribution as defined by Kipping (2013) with shape parameters.
    """
    a = 0.867
    b = 3.03
    theta_scaled = beta.ppf(theta, a, b)
    return theta_scaled


def rayleigh(theta, sigma):
    """
    Transforms a uniform variable into a value sampled from a truncated Rayleigh distribution.

    Parameters
    ----------
    theta : float
        Uniformly distributed variable [0, 1] from the unit hypercube.
    sigma : float
        Scale parameter of the Rayleigh distribution.

    Returns
    -------
    float
        A value sampled from a Rayleigh distribution.
    """
    theta_scaled = sigma * sqrt(-2 * log(1 - theta))
    return theta_scaled


def uniform(theta, a, b):
    """
    Transforms a uniform variable into another uniformly distributed variable over [a, b].

    Parameters
    ----------
    theta : float
        Uniformly distributed variable [0, 1] from the unit hypercube.
    a : float
        Lower bound of the target uniform distribution.
    b : float
        Upper bound of the target uniform distribution.

    Returns
    -------
    float
        A value uniformly distributed between [a, b].
    """
    theta_scaled = a + (b - a) * theta
    return theta_scaled
