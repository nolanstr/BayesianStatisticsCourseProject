import numpy as np

def get_sufficient_statistics_gamma_alpha(x):
    n = x.shape[0]
    p = np.prod(x)
    return n, p


def get_sufficient_statistics_gamma_beta(x):
    n = x.shape
    sum_x = np.sum(x)
    return n, sum_x


def get_sufficient_statistics_lognormal_precision(x, mu):
    n = x.shape[0]
    SS = np.sum(np.square(np.log(x) - mu))
    return n, SS


def get_sufficient_statistics_lognormal_mu(x):
    n = x.shape[0]
    x_bar = np.sum(np.log(x)) / n
    return n, x_bar


def get_sufficient_statistics_normal_mu(x):
    n = x.shape[0]
    x_bar = np.sum(x) / n
    return n, x_bar


def get_sufficient_statistics_normal_precision(x, mu):
    n = x.shape[0]
    SS = np.sum(np.square(x - mu))
    return n, SS
