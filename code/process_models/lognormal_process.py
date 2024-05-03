import numpy as np
from scipy.stats import norm, gamma

from process_models.model import Model

class LogNormalNormalMean(Model):
    """
    Model for LogNormal Sampling Model with Normal Prior for Mean.
    This model is intended for within group sampling.
    """

    def __init__(self, priors, n_groups, init_current_state=False):
        super().__init__(priors, n_dependencies=2,
                         init_current_state=init_current_state)
        self._n_groups = n_groups
        if isinstance(init_current_state, bool):
            if not init_current_state:
                self.samples = np.zeros(n_groups)
        self._dependencies_keys = ["dep", "x"]
        m, p = self.get_priors()
        self._distribution = norm(loc=m, scale=np.sqrt(p))
        self._distribution_df = self._distribution.pdf

    def sample_full_conditional(self):
        self._check_for_dependencies()
        rho, x = self.get_dependencies()
        m, p = self.get_priors()
        sample = np.empty(self._n_groups)

        for i, (rho_i, x_i) in enumerate(zip(rho, x)):
            n, x_bar = self.get_sufficient_statistics(x_i)
            p_prime = (p + (n * rho_i))
            m_prime = ((m * p) + (n * rho_i * x_bar)) / (p_prime)
            sample[i] = norm(loc=m_prime, scale=np.sqrt(1 / p_prime)).rvs()
        self.samples = np.vstack((self.samples, sample))

    def get_dependencies(self):
        rho = self.dependencies["dep"].current_state
        x = self.dependencies["x"]
        return rho, x

    def get_priors(self):
        m = self.priors["m"].current_state
        p = self.priors["p"].current_state
        return m, p

    def get_sufficient_statistics(self, x):
        n = x.shape[0]
        x_bar = np.sum(np.log(x)) / n
        return n, x_bar


class LogNormalInvGammaVariance(Model):
    """
    Model for LogNormal Sampling Model with InvGamma Prior for Variance.
    """

    def __init__(self, priors, init_current_state=False):
        #super().__init__(priors, n_dependencies=2,
        #                 init_current_state=init_current_state)
        raise NotImplementedError

    def sample_full_conditional(self):
        self._check_for_dependencies()
        self.samples = np.append(self.samples, sample)

    def get_dependencies(self):
        return None

    def get_priors(self):
        return None

    def get_sufficient_statistics(self, x):
        return None


class LogNormalGammaPrecision(Model):
    """
    Model for LogNormal Sampling Model with Gamma Prior for Precision.
    """

    def __init__(self, priors, n_groups, init_current_state=False):
        super().__init__(priors, n_dependencies=2,
                         init_current_state=init_current_state)
        self._n_groups = n_groups
        if isinstance(init_current_state, bool):
            if not init_current_state:
                self.samples = np.zeros((0,n_groups))
        self._dependencies_keys = ["dep", "x"]
        alpha, beta = self.get_priors()
        self._distribution = gamma(a=alpha, scale=1/beta)
        self._distribution_df = self._distribution.pdf

    def sample_full_conditional(self):
        self._check_for_dependencies()

        mu, x = self.get_dependencies()
        alpha, beta = self.get_priors()
        sample = np.empty(self._n_groups)
        for i, (mu_i, x_i) in enumerate(zip(mu, x)):
            n, SS = self.get_sufficient_statistics(x_i, mu_i)
            alpha_prime = alpha + (n / 2)
            beta_prime = beta + (SS / 2)
            sample[i] = gamma(a=alpha_prime, scale=1/beta_prime).rvs()

        self.samples = np.vstack((self.samples, sample))

    def get_dependencies(self):
        mu = self.dependencies["dep"].current_state
        x = self.dependencies["x"]
        return mu, x

    def get_priors(self):
        alpha = self.priors["alpha"].current_state
        beta = self.priors["beta"].current_state
        return alpha, beta

    def get_sufficient_statistics(self, x, mu):
        n = x.shape[0]
        SS = np.sum(np.square(np.log(x) - mu))
        return n, SS
