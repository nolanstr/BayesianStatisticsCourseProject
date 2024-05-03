import numpy as np
from tqdm import tqdm


class HierarchicalNCV:
    """
    Hierarchical Bayesian Model with Non-Constant Variance.
    """

    def __init__(self, mu_BG, rho_BG, alpha_BG, beta_BG, mu_WG, rho_WG):
        """
                    Between Group Models:
        mu_BG: Between Group Mean of Means
        rho_BG: Between Group Precision of Means
        alpha_BG: Between Group Alpha (Scale) of Precisions
        beta_BG: Between Group Beta (1/Rate) of Precisions

                    Within Group Models:
        mu_WG: Within Group Mean
        rho_WG: Within Group Precision
        """
        self.models = {
            "mu_BG": mu_BG,
            "rho_BG": rho_BG,
            "alpha_BG": alpha_BG,
            "beta_BG": beta_BG,
            "mu_WG": mu_WG,
            "rho_WG": rho_WG,
        }
        self._check_models()
    
    
    def run_gibbs_sampler(self, S=1000):

        self.models["rho_WG"].sample_full_conditional()

        for s in tqdm(range(S)):
            self.models["mu_BG"].sample_full_conditional()
            self.models["rho_BG"].sample_full_conditional()
            self.models["alpha_BG"].sample_full_conditional()
            self.models["beta_BG"].sample_full_conditional()
            self.models["mu_WG"].sample_full_conditional()
            self.models["rho_WG"].sample_full_conditional()
    
    def _check_models(self):
        init_models = []
        for key in self.models.keys():
            if key != "rho_WG":
                if self.models[key].samples.shape[0] == 0:
                    init_models.append(key)
        if len(init_models) > 0:
            raise ValueError("Models Need Initialization: {init_models}")

class BaseHierarchicalNCV:
    """
    Hierarchical Bayesian Model with Non-Constant Variance.
    """

    def __init__(self, models):
        """
            Models are passed in the order in which they will be sampled
            from, i.e.
            for s in range(S):
                for n in range(len(models)):
                    sample full conditional of model n

        """
        self.models = models 
        self._check_models()
    
    
    def run_gibbs_sampler(self, S=1000):

        self.models[-1].sample_full_conditional()

        for s in tqdm(range(S)):
            for model in self.models:
                model.sample_full_conditional()
    
    def _check_models(self):
        init_models = []
        for i in range(len(self.models)-1):
            if self.models[i].samples.shape[0] == 0:
                init_models.append(i+1)
        if len(init_models) > 0:
            raise ValueError("Models Need Initialization: {init_models}")
