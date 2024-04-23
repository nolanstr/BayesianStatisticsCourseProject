from scipy.stats import norm, gamma
from util import *


def sample_mu_i(x_i, rho_i, m, p):
    """
    x_i: log-normally distributed data (within group)
    rho_i: precision of i^th group (within group)
    m: mean parameter (between group)
    p: precision (between group)
    """
    n, x_bar = get_sufficient_statistics_lognormal_mu(x_i)
    p_prime = p + (n * rho_i)
    m_prime = ((m * p) + (n * rho_i * x_bar)) / (p_prime)
    mu_i = norm(loc=m_prime, scale=np.sqrt(1 / p_prime)).rvs()
    return mu_i

def sample_all_mu_i(x, rho, m, p):
    """
    Samples mu_i for all i groups
    """
    samples = np.empty(len(x))
    for i, (x_i, rho_i) in enumerate(zip(x, rho)):
        samples[i] = sample_mu_i(x_i, rho_i, m, p)
    return samples

def sample_rho_i(x_i, mu_i, alpha, beta):
    """
    x_i: log-normally distributed data (within group)
    mu_i: mean of i^th group (within group)
    alpha: alpha parameter (between group)
    beta: beta parameter (between group)
    """
    n, SS = get_sufficient_statistics_lognormal_precision(x_i, mu_i)
    alpha_prime = alpha + (n / 2)
    beta_prime = beta + (SS / 2)
    rho_i = gamma(a=alpha_prime, scale=beta_prime).rvs()
    return rho_i

def sample_all_rho_i(x, mu, alpha, beta):
    """
    Samples rho_i for all i groups
    """
    samples = np.empty(len(x))
    for i, (x_i, mu_i) in enumerate(zip(x, mu)):
        samples[i] = sample_rho_i(x_i, mu_i, alpha, beta)
    return samples
