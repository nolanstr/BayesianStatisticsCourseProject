import numpy as np
from scipy.special import gamma as gamma_function, loggamma
from scipy.stats import norm, gamma, invgamma, geom
import matplotlib.pyplot as plt
import seaborn as sns

from process_models.model import Model

class GammaGeometricScale(Model):
    """
    Model for Gamma Sampling Model with Geometric Prior for Scale (alpha).
    Model proposed by Hoff (more intuitive)
    """

    def __init__(self, priors, init_current_state=False, 
                                alpha_domain=np.arange(1, 51)):
        super().__init__(
            priors, n_dependencies=2, init_current_state=init_current_state
        )
        self._alpha = alpha_domain
        self._dependencies_keys = ["dep", "x"]
        a = self.get_priors()
        self._distribution = geom(p=a)
        self._distribution_df = lambda x: np.exp(-a*x) 

    def sample_full_conditional(self):
        self._check_for_dependencies()
        beta, x = self.get_dependencies()

        alpha = self.get_priors()
        likes = self.compute_likelihoods(beta, x)
        priors = self._distribution_df(self._alpha)
        terms = likes * priors
        weights = terms / terms.sum()
        weights = weights / weights.sum()
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
        beta = self.dependencies["dep"].current_state
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

    def plot_prior_vs_posterior(self, burnin=50, show=False):
        filename = f"prior_vs_post/alpha"
        fig, ax = plt.subplots()
        counts = np.round(self._distribution_df(self._alpha)*1E6).astype(int)
        prior_X = np.hstack([np.ones(count)*_alpha \
                        for count, _alpha in zip(counts, self._alpha)])
        sns.histplot(x=self.samples[burnin:].flatten(),
                    ax=ax,
                    label="Posterior",
                    color="orange",
                    stat="density")
        sns.histplot(x=prior_X, 
                     ax=ax, 
                     label="Prior",
                     stat="density")
        ax.legend()
        label = r"$\alpha$"
        ax.set_xlabel(label)
        fig.suptitle(f"Posterior vs. Prior for {label}")
        lgd, keys = ax.get_legend_handles_labels()
        d = dict(zip(keys, lgd))
        ax.legend(d.values(), d.keys())
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.clf()


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
        self._dependencies_keys = ["dep", "x"]
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
        alpha = self.dependencies["dep"].current_state
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

    def plot_prior_vs_posterior(self, burnin=50, show=False):
        filename = f"prior_vs_post/beta"
        fig, ax = plt.subplots()
        X = np.linspace(*self._distribution.interval(0.95), 1000)
        sns.kdeplot(x=X, 
                    weights=self._distribution_df(X), 
                    ax=ax, 
                    label="Prior", 
                    fill=True,
                    cut=0)
        sns.kdeplot(x=self.samples[burnin:].flatten(), 
                    ax=ax, 
                    label="Posterior", 
                    fill=True,
                    cut=0)
        ax.legend()
        label = r"$\beta$"
        ax.set_xlabel(label)
        fig.suptitle(f"Posterior vs. Prior for {label}")
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.clf()
