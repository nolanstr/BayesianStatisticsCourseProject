
import numpy as np
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import seaborn as sns

from process_models.model import Model

class NormalNormalMean(Model):
    """
    Model for Normal Sampling Model with Normal Prior for Mean.
    """

    def __init__(self, priors, init_current_state=False):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._dependencies_keys = ["dep", "x"]
        mu_0, rho_0 = self.get_priors()
        self._distribution = norm(loc=mu_0, scale=np.sqrt(1/rho_0))
        self._distribution_df = self._distribution.pdf

    def sample_full_conditional(self):
        self._check_for_dependencies()
        rho, x = self.get_dependencies()
        n, x_bar = self.get_sufficient_statistics(x)
        mu_0, var_0 = self.get_priors()
        rho_0 = 1 / var_0

        p_prime = rho_0 + n * rho
        m_prime = ((mu_0 * rho_0) + (n * rho * x_bar)) / p_prime
        mu = norm(loc=m_prime, scale=np.sqrt(1 / p_prime)).rvs()
        if isinstance(mu, np.float64):
            mu = np.array([mu])
        self.samples = np.vstack((self.samples, mu))

    def get_dependencies(self):

        if isinstance(self.dependencies["dep"], NormalInvGammaVariance):
            raise ValueError("NormalInvGammaVariance Not Supported Currently.")
            rho = 1 / self.dependencies["dep"].current_state
        elif isinstance(self.dependencies["dep"], NormalGammaPrecision):
            rho = self.dependencies["dep"].current_state
        else:
            raise ValueError("Dependency Not Supported.")
        x = self.dependencies["x"].current_state
        return rho, x

    def get_priors(self):
        mu_0 = self.priors["mu_0"]
        if "rho_0" in self.priors.keys():
            rho_0 = self.priors["rho_0"]
        else:
            var_0 = self.priors["var_0"]
            rho_0 = 1 / var_0

        if not isinstance(mu_0, float):
            mu_0 = mu_0.current_state
        if not isinstance(rho_0, float):
            rho_0 = rho_0.current_state

        return mu_0, rho_0

    def get_sufficient_statistics(self, x):
        n = x.shape[0]
        x_bar = np.sum(x) / n
        return n, x_bar

    def plot_prior_vs_posterior(self, burnin=50, show=False):
        filename = f"prior_vs_post/mu"
        fig, ax = plt.subplots()
        interval = self._distribution.interval(0.95)
        X = np.linspace(interval[0], interval[1], 1000)
        weights = self._distribution_df(X)
        sns.kdeplot(x=X, weights=weights, ax=ax, label="Prior", fill=True,
                    cut=0)
        sns.kdeplot(x=self.samples[burnin:].flatten(), 
                    ax=ax, 
                    label="Posterior", 
                    fill=True,
                    cut=0)
        ax.legend()
        label = r"$\mu$"
        ax.set_xlabel(label)
        fig.suptitle(f"Posterior vs. Prior for {label}")
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.clf()


class NormalGammaPrecision(Model):
    """
    Model for Normal Sampling Model with Gamma Prior for Precision.
    """

    def __init__(self, priors, init_current_state=False):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._dependencies_keys = ["dep", "x"]
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
        if isinstance(rho, np.float64):
            rho = np.array([rho])
        self.samples = np.vstack((self.samples, rho))

    def get_dependencies(self):
        mu = self.dependencies["dep"].current_state
        x = self.dependencies["x"].current_state
        return mu, x

    def get_priors(self):
        alpha = self.priors["alpha"]
        beta = self.priors["beta"]
        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            alpha = alpha.current_state
        if not (isinstance(beta, float) or isinstance(beta, int)):
            beta = beta.current_state

        return alpha, beta

    def get_sufficient_statistics(self, x, mu):
        n = x.shape[0]
        SS = np.sum(np.square(x - mu))
        return n, SS

    def plot_prior_vs_posterior(self, burnin=50, show=False):
        filename = f"prior_vs_post/rho"
        fig, ax = plt.subplots()
        interval = self._distribution.interval(0.95)
        x = np.linspace(interval[0], interval[1], 1000)
        weights = self._distribution_df(x)
        sns.kdeplot(x=x, weights=weights, ax=ax, label="prior", fill=True,
                    cut=0)
        sns.kdeplot(x=self.samples[burnin:].flatten(), 
                    ax=ax, 
                    label="posterior", 
                    fill=True,
                    cut=0)
        ax.legend()
        label = r"$\rho$"
        ax.set_xlabel(label)
        fig.suptitle(f"posterior vs. prior for {label}")
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.clf()

class NormalInvGammaVariance(Model):
    """
    Model for Normal Sampling Model with InvGamma Prior for Variance.
    """

    def __init__(self, priors, init_current_state=False):
        raise NotImplementedError("NormalInvGammaVariance Not Supported Currently.")
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._dependencies_keys = ["dep", "x"]
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
        mu = self.dependencies["dep"].current_state
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

