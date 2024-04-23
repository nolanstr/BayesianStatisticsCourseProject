import numpy as np
from scipy.special import gamma as gamma_function
from scipy.stats import norm, gamma, invgamma

from util import *


def sample_mu(rho, m, p, x):
    """
    Prior mean for between group mean

    rho: Precision
    m: Prior mean
    p: Prior precision
    x: Mean for each group (vector, mu's)
    """
    n, x_bar = get_sufficient_statistics_normal_mu(x)
    p_prime = p + n * rho
    m_prime = ((m * p) + (n * rho * x_bar)) / p_prime
    mu = norm(loc=m_prime, scale=np.sqrt(1 / p_prime)).rvs()
    return mu


def sample_rho(mu, alpha, beta, x):
    """
    Prior precision for between group mean

    mu: Mean
    alpha: Prior alpha parameter
    beta: Prior beta parameter
    x: Mean for each group (vector)
    """

    n, SS = get_sufficient_statistics_normal_precision(x, mu)
    alpha_prime = alpha + (n / 2)
    beta_prime = beta + (SS / 2)
    rho = gamma(a=alpha_prime, scale=beta_prime).rvs()
    return rho


def sample_alpha(beta, a, b, c, x, alpha_domain=np.arange(1, 21)):
    """
    Prior alpha (geometric prior) for between group variance

    beta: beta paramater
    a: prior parameter
    b: prior  parameter
    c: prior  parameter
    x: Precision for each group (vector)
    alpha_domain: Range of values of alpha to evaluate over
    """
    alpha_domain = alpha_domain.astype(int)
    n, p = get_sufficient_statistics_gamma_alpha(x)
    a_prime = a*p
    b_prime = b + n
    c_prime = c + n
    num = np.power(alpha_domain, a_prime-1) * \
                    np.power(beta, alpha_domain*c)
    den = np.power(gamma_function(alpha_domain), b_prime)
    terms = num / den
    weights = terms / terms.sum()
    alpha = np.random.choice(alpha_domain, p=weights)
    return alpha


def sample_beta(alpha, a, b, x):
    """
    Prior beta for between group variance

    alpha: alpha paramater
    a: prior alpha parameter
    b: prior  beta parameter
    x: Precision for each group (vector)
    """
    n, sum_x = get_sufficient_statistics_gamma_beta(x)
    a_prime = (alpha * n) + a
    b_prime = b / (1 + (b*sum_x))
    beta = invgamma(a=a_prime, scale=b_prime).rvs()
    return beta 
