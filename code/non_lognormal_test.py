import numpy as np
import pandas as pd

from hierarchical_models.nonconstant_variance import BaseHierarchicalNCV

from process_models.normal_process import *
from process_models.gamma_process import *
from process_models.lognormal_process import *


from plot_distributions import (visualize_lognormal_distributions,
                                plot_prior_vs_posterior)

class DataModel:
    def __init__(self, x):
        self.current_state = x

if __name__ == "__main__":
    
    # Initialize data
    
    df = pd.read_csv("../single_scan.csv")
    x = []
    variant_ids = np.unique(df["alpha_variant"])
    for variant_id in variant_ids:
        variant_df = df.loc[df["alpha_variant"] == variant_id]
        equivalent_diameter = variant_df["equivalent_diameter"].to_numpy()
        equivalent_radius = equivalent_diameter / 2
        # Transform data to normal distribution
        x.append(np.log(equivalent_radius)) 
    
    x = DataModel(x) # Gives data a current state

    # Prior parameters for between group models
    ns = np.array([len(x_) for x_ in x])
    sample_means = np.array([x_.mean() for x_ in x])
    sample_vars = np.array([x_.var() for x_ in x])

    mean_of_means = np.mean(sample_means)
    var_of_means = np.var(sample_means)

    mean_of_vars = np.mean(sample_vars)
    var_of_vars = np.var(sample_vars)
    
    # Initialize Models

    mu_BG_priors = {"mu_0": mean_of_means, "rho_0": 1}
    mu_BG = NormalNormalMean(priors=mu_BG_priors, 
                             init_current_state=0.)

    rho_BG_priors = {"alpha": 1, "beta": 1/20}
    rho_BG = NormalGammaPrecision(priors=rho_BG_priors, 
                                  init_current_state=1) 
    alpha_BG_priors = {"a": 1/10}
    alpha_BG = GammaGeometricScaleHoff(priors=alpha_BG_priors,
                                   init_current_state=2)
    beta_BG_priors = {"a": 1, "b": 100}
    beta_BG = GammaGammaBeta(priors=beta_BG_priors,
                             init_current_state=2)

    mu_WG_priors = {"mu_0": mu_BG, "rho_0": rho_BG}
    x_means = np.array([x_i.mean() for x_i in x])
    x_vars = np.array([x_i.var() for x_i in x])
    mu_WG_current_state = x_means
    mu_WG = NormalNormalMean(priors=mu_WG_priors,
                                init_current_state=mu_WG_current_state)
    
    rho_WG_priors = {"alpha":alpha_BG, "beta":beta_BG}
    rho_WG = NormalGammaPrecision(priors=rho_WG_priors)
    
    # Set Dependencies

    mu_BG.dependencies = {"dep":rho_BG, "x":mu_WG}
    rho_BG.dependencies = {"dep":mu_BG, "x":mu_WG}
    alpha_BG.dependencies = {"dep":rho_BG, "x":rho_WG}
    beta_BG.dependencies = {"dep":rho_BG, "x":rho_WG}
    mu_WG.dependencies = {"dep":rho_WG, "x":x}
    rho_WG.dependencies = {"dep":mu_WG, "x":x}
    
    models = [mu_BG, rho_BG, alpha_BG, beta_BG, mu_WG, rho_WG]

    model = BaseHierarchicalNCV(models)
    model.run_gibbs_sampler(S = 1000)
    mu_samples = model.models[-2].samples
    rho_samples = model.models[-1].samples
    var_samples = 1 / rho_samples
    print("Work on visualization code later!!!!")
    visualize_lognormal_distributions(x, mu_samples, rho_samples)
    labels = [r"$\mu$", r"$\tau^{2}$", r"$\alpha$", r"$\beta$"]
    file_tags = ["mu_BG", "rho_BG", "alpha_BG", "beta_BG"]
    for i, (label_i, tag_i) in enumerate(zip(labels, file_tags)):

        import pdb;pdb.set_trace()
        model_for_plots = model.models[i]
        plot_prior_vs_posterior(model_for_plots, label_i, tag_i, burnin=50)

    import pdb;pdb.set_trace()
