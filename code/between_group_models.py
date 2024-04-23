import numpy as np
from scipy.special import gamma as gamma_function, loggamma
from scipy.stats import norm, gamma, invgamma, geom

from model import Model


# Normal Sampling Model Models
class NormalNormalMean(Model):
    """
    Model for Normal Sampling Model with Normal Prior for Mean.
    """

    def __init__(self, priors, init_current_state=False, dependency_key="var"):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._dependencies_keys = [dependency_key, "x"]
        mu_0, rho_0 = self.get_priors()
        self._distribution = norm(loc=mu_0, scale=np.sqrt(1/rho_0))
        self._distribution_df = self._distribution.pdf

    def sample_full_conditional(self):
        self._check_for_dependencies()
        rho, x = self.get_dependencies()
        n, x_bar = self.get_sufficient_statistics(x)
        mu_0, rho_0 = self.get_priors()

        p_prime = rho_0 + n * rho
        m_prime = ((mu_0 * rho_0) + (n * rho * x_bar)) / p_prime
        mu = norm(loc=m_prime, scale=np.sqrt(1 / p_prime)).rvs()

        self.samples = np.append(self.samples, mu)

    def get_dependencies(self):
        if self._dependencies_keys[0] == "var":
            var = self.dependencies["var"].current_state
            rho = 1 / var
        else:
            rho = self.dependencies["rho"].current_state
        x = self.dependencies["x"].current_state
        return rho, x

    def get_priors(self):
        mu_0 = self.priors["mu_0"]
        rho_0 = self.priors["rho_0"]
        return mu_0, rho_0

    def get_sufficient_statistics(self, x):
        n = x.shape[0]
        x_bar = np.sum(x) / n
        return n, x_bar


class NormalGammaPrecision(Model):
    """
    Model for Normal Sampling Model with InvGamma Prior for Precision.
    This model is to be used with a mean dependency.
    """

    def __init__(self, priors, init_current_state=False):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._dependencies_keys = ["mu", "x"]
        alpha, beta = self.get_priors()
        self._distribution = gamma(a=alpha, scale=1/beta)
        self._distribution_df = self._distribution.pdf

    def sample_full_conditional(self):
        self._check_for_dependencies()
        mu, x = self.get_dependencies()
        n, SS = self.get_sufficient_statistics(x, mu)

        alpha, beta = self.get_priors()

        alpha_prime = alpha + (n / 2)
        beta_prime = beta + (SS / 2)
        rho = gamma(a=alpha_prime, scale=1/beta_prime).rvs()

        self.samples = np.append(self.samples, rho)

    def get_dependencies(self):
        mu = self.dependencies["mu"].current_state
        x = self.dependencies["x"].current_state
        return mu, x

    def get_priors(self):
        alpha = self.priors["alpha"]
        beta = self.priors["beta"]
        return alpha, beta

    def get_sufficient_statistics(self, x, mu):
        n = x.shape[0]
        SS = np.sum(np.square(x - mu))
        return n, SS


# Gamma Sampling Model
class GammaGeometricScaleFink(Model):
    """
    Model for Gamma Sampling Model with Geometric Prior for Scale (alpha).
    Model proposed by Fink (less intuitive)
    """

    def __init__(self, priors, init_current_state=False, alpha_domain=np.arange(1, 21)):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._alpha_domain = alpha_domain
        self._dependencies_keys = ["beta", "x"]

    def sample_full_conditional(self):
        self._check_for_dependencies()
        beta, x = self.get_dependencies()
        n, p = self.get_sufficient_statistics(x)
        a, b, c = self.get_priors()

        a_prime = a * p
        b_prime = b + n
        c_prime = c + n
        num = np.power(self._alpha_domain, a_prime - 1) * np.power(
            beta, self._alpha_domain * c
        )
        den = np.power(gamma_function(self._alpha_domain), b_prime)
        terms = num / den
        weights = terms / terms.sum()
        alpha = np.random.choice(self._alpha_domain, p=weights)

        self.samples = np.append(self.samples, alpha)

    def get_dependencies(self):
        beta = self.dependencies["beta"].current_state
        x = self.dependencies["x"].current_state
        return beta, x

    def get_priors(self):
        a = self.priors["a"]
        b = self.priors["b"]
        c = self.priors["c"]
        return a, b, c

    def get_sufficient_statistics(self, x):
        n = x.shape[0]
        p = np.prod(x)
        return n, p

# Gamma Sampling Model
class GammaGeometricScaleHoff(Model):
    """
    Model for Gamma Sampling Model with Geometric Prior for Scale (alpha).
    Model proposed by Hoff (more intuitive)
    """

    def __init__(self, priors, init_current_state=False, 
                                alpha_domain=np.arange(1, 6)):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._alpha = alpha_domain
        self._dependencies_keys = ["beta", "x"]
        a = self.get_priors()
        self._distribution = geom(p=a)
        self._distribution_df = lambda x: np.exp(-a*x) 

    def sample_full_conditional(self):
        self._check_for_dependencies()
        beta, x = self.get_dependencies()
        sigma2 = 1/x
        s20 = 1 / beta 
        x = self._alpha

        n, p, s = self.get_sufficient_statistics(x)
        m = n
        alpha = self.get_priors()
        terms = m * (0.5*x*np.log(s20*x/2) - loggamma(x/2) ) +\
                (x/2 - 1) * sum(np.log(1 / sigma2)) + \
                -x * (alpha + .5*s20*sum(1/sigma2))

        if np.sum(terms) == 0.:
            terms = np.ones_like(terms)
        weights = np.exp(terms - terms.max()) 
        weights = weights / np.sum(weights)
        alpha = np.random.choice(self._alpha, p=weights)

        self.samples = np.append(self.samples, alpha)
    
    def compute_priors(self, a):
        priors = np.power(1-a, self._alpha) * a
        return priors

    def compute_likelihoods(self, beta, rho):
        likes = np.array([
            np.prod(gamma(a=alpha, scale=1/beta).pdf(rho)) \
                    for alpha in self._alpha])
        return likes
    
    def get_dependencies(self):
        beta = self.dependencies["beta"].current_state
        x = self.dependencies["x"].current_state
        return beta, x

    def get_priors(self):
        a = self.priors["a"]
        return a

    def get_sufficient_statistics(self, x):
        n = x.shape[0]
        p = np.prod(x)
        s = np.sum(x)
        return n, p, s


class GammaGammaRate(Model):
    """
    Model for Gamma Sampling Model with Geometric Prior for rate (1/beta).
    """

    def __init__(self, priors, init_current_state=False):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        raise NotImplementedError

    def sample_full_conditional(self):
        self._check_for_dependencies()
        self.samples = np.append(self.samples, sample)

    def get_sufficient_statistics(self, x):
        return sufficient_statistics


class GammaGammaBeta(Model):
    """
    Model for Gamma Sampling Model with Geometric Prior for beta.
    Beta = 1 / var
    """

    def __init__(self, priors, init_current_state):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._dependencies_keys = ["alpha", "x"]
        a, b = self.get_priors()
        self._distribution = gamma(a=a, scale=1/b)
        self._distribution_df = self._distribution.pdf

    def sample_full_conditional(self):
        self._check_for_dependencies()
        alpha, x = self.get_dependencies()
        n, sum_x = self.get_sufficient_statistics(x)
        a, b = self.get_priors()

        a_prime = (alpha * n) + a
        b_prime = b / (1 + (b * sum_x))
        beta = 1 / gamma(a=a_prime, scale=1/b_prime).rvs()

        self.samples = np.append(self.samples, beta)

    def get_dependencies(self):
        alpha = self.dependencies["alpha"].current_state
        x = self.dependencies["x"].current_state
        return alpha, x

    def get_priors(self):
        a = self.priors["a"]
        b = self.priors["b"]
        return a, b

    def get_sufficient_statistics(self, x):
        n = x.shape[0]
        sum_x = np.sum(x)
        return n, sum_x
