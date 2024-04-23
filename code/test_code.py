import numpy as np
import pandas as pd

from hierarchical_models import HierarchicalNCV
from between_group_models import (
    NormalNormalMean,
    NormalGammaPrecision,
    GammaGeometricScaleHoff,
    GammaGammaBeta,
)
from within_group_models import LogNormalNormalMean, LogNormalGammaPrecision

from plot_distributions import (visualize_lognormal_distributions,
                                plot_prior_vs_posterior)

if __name__ == "__main__":
    
    # Initialize data
    
    df = pd.read_csv("../single_scan.csv")
    x = []
    variant_ids = np.unique(df["alpha_variant"])
    for variant_id in variant_ids:
        variant_df = df.loc[df["alpha_variant"] == variant_id]
        equivalent_diameter = variant_df["equivalent_diameter"].to_numpy()
        equivalent_radius = equivalent_diameter / 2
        x.append(equivalent_radius)
    
    n_groups = len(x)
    # Prior parameters for between group models
    ns = np.array([len(x_) for x_ in x])
    sample_means = np.array([x_.mean() for x_ in x])
    sample_vars = np.array([x_.var() for x_ in x])

    mean_of_means = np.mean(sample_means)
    var_of_means = np.var(sample_means)

    mean_of_vars = np.mean(sample_vars)
    var_of_vars = np.var(sample_vars)
    
    # Initialize Models

    mu_BG_priors = {"mu_0": mean_of_means, "rho_0": 10}
    mu_BG = NormalNormalMean(priors=mu_BG_priors, 
                             init_current_state=0.,
                             dependency_key="rho")

    rho_BG_priors = {"alpha": 1, "beta": 1/20}
    rho_BG = NormalGammaPrecision(priors=rho_BG_priors, 
                                  init_current_state=1) 
    alpha_BG_priors = {"a": 1}
    alpha_BG = GammaGeometricScaleHoff(priors=alpha_BG_priors,
                                   init_current_state=2)
    beta_BG_priors = {"a": 1, "b": 100}
    beta_BG = GammaGammaBeta(priors=beta_BG_priors,
                             init_current_state=2)

    mu_WG_priors = {"m": mu_BG, "p": rho_BG}
    x_means = np.array([x_i.mean() for x_i in x])
    x_vars = np.array([x_i.var() for x_i in x])
    mu_WG_current_state = np.log(x_means)
    mu_WG = LogNormalNormalMean(priors=mu_WG_priors,
                                n_groups=n_groups,
                                init_current_state=mu_WG_current_state)
    
    rho_WG_priors = {"alpha":alpha_BG, "beta":beta_BG}
    rho_WG = LogNormalGammaPrecision(priors=rho_WG_priors,
                                     n_groups=n_groups)
    
    # Set Dependencies

    mu_BG.dependencies = {"rho":rho_BG, "x":mu_WG}
    rho_BG.dependencies = {"mu":mu_BG, "x":mu_WG}
    alpha_BG.dependencies = {"beta":rho_BG, "x":rho_WG}
    beta_BG.dependencies = {"alpha":rho_BG, "x":rho_WG}
    mu_WG.dependencies = {"rho":rho_WG, "x":x}
    rho_WG.dependencies = {"mu":mu_WG, "x":x}

    model = HierarchicalNCV(mu_BG, 
                            rho_BG, 
                            alpha_BG, 
                            beta_BG, 
                            mu_WG, 
                            rho_WG)
    model.run_gibbs_sampler(S = 100)
    mu_samples = model.models["mu_WG"].samples
    rho_samples = model.models["rho_WG"].samples
    var_samples = 1 / rho_samples
    visualize_lognormal_distributions(x, mu_samples, var_samples)
    labels = [r"$\mu$", r"$\tau^{2}$", r"$\alpha$", r"$\beta$"]
    file_tags = ["mu_BG", "rho_BG", "alpha_BG", "beta_BG"]
    for (label_i, tag_i) in zip(labels, file_tags):

        model_for_plots = model.models[tag_i]
        plot_prior_vs_posterior(model_for_plots, label_i, tag_i, burnin=50)

    import pdb;pdb.set_trace()
