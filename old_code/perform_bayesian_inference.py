import numpy as np
import pandas as pd

from within_group_sample import *
from between_group_sample import *


def gibbs_sampler(data, 
                  mu_priors, 
                  rho_priors, 
                  alpha_priors, 
                  beta_priors, 
                  S):
    """
    data: Lognormally distributed data for n groups
    mu_priors: Priors for the mean of the between group means
    rho_priors: Priors for the precision of the between group means
    alpha_priors: Priors for the alpha of the between group precision
    beta_priors: Priors for the beta of the between group precision
    S: Number of samples
    """

    n_groups = len(data)
    (
        mu_samples,
        rho_samples,
        alpha_samples,
        beta_samples,
        mu_i_samples,
        rho_i_samples,
    ) = initialize_samples(data, 
                           mu_priors, 
                           rho_priors, 
                           alpha_priors, 
                           beta_priors, 
                           S)
    mu_i_samples[0, :] = np.array([data_i.mean() for data_i in data])
    rho_i_samples[0, :] = sample_all_rho_i(data, 
                                           mu_i_samples[0],
                                           alpha_samples[0],
                                           beta_samples[0]) 
    for s in range(S):
        
        mu_samples[s+1] = sample_mu(rho_samples[s], 
                                    mu_priors["m"],
                                    mu_priors["p"],
                                    mu_i_samples[s])
        rho_samples[s+1] = sample_rho(mu_samples[s+1],
                                      rho_priors["alpha"],
                                      rho_priors["beta"],
                                      mu_i_samples[s])
        alpha_samples[s+1] = sample_alpha(beta_samples[s],
                                          alpha_priors["a"],
                                          alpha_priors["b"],
                                          alpha_priors["c"],
                                          rho_i_samples[s])
        beta_samples[s+1] = sample_beta(alpha_samples[s+1],
                                        beta_priors["a"],
                                        beta_priors["b"],
                                        rho_i_samples[s])
        mu_i_samples[s+1] = sample_all_mu_i(data, 
                                       rho_i_samples[s],
                                       mu_samples[s+1],
                                       rho_samples[s+1])
        rho_i_samples[s+1] = sample_all_rho_i(data,
                                              mu_i_samples[s+1],
                                              alpha_samples[s+1],
                                              beta_samples[s+1])
    
    return (mu_samples, 
            rho_samples, 
            alpha_samples, 
            beta_samples, 
            mu_i_samples, 
            rho_i_samples)


def initialize_samples(data, mu_priors, rho_priors, alpha_priors, beta_priors, S):
    n_groups = len(data)

    mu_samples = np.empty((S + 1, 1))
    mu_samples[0, :] = mu_priors["sample"]

    rho_samples = np.empty((S + 1, 1))
    rho_samples[0, :] = rho_priors["sample"]

    alpha_samples = np.empty((S + 1, 1))
    alpha_samples[0, :] = alpha_priors["sample"]

    beta_samples = np.empty((S + 1, 1))
    beta_samples[0, :] = beta_priors["sample"]

    mu_i_samples = np.empty((S + 1, n_groups))
    rho_i_samples = np.empty((S + 1, n_groups))

    return (
        mu_samples,
        rho_samples,
        alpha_samples,
        beta_samples,
        mu_i_samples,
        rho_i_samples,
    )


if __name__ == "__main__":
    df = pd.read_csv("single_scan.csv")
    x = []
    variant_ids = np.unique(df["alpha_variant"])
    for variant_id in variant_ids:
        variant_df = df.loc[df["alpha_variant"] == variant_id]
        equivalent_diameter = variant_df["equivalent_diameter"].to_numpy()
        equivalent_radius = equivalent_diameter / 2
        x.append(equivalent_radius)
    mu_priors = {"sample": 0, "m": 0, "p": 1}
    rho_priors = {"sample": 1, "alpha": 1, "beta": 1}
    alpha_priors = {"sample": 1, "a": 1e-57, "b": 1.0, "c": 1.25}
    beta_priors = {"sample": 2, "a": 1, "b": 1}
    S = 1000
    (mu_samples, 
     rho_samples, 
     alpha_samples, 
     beta_samples, 
     mu_i_samples, 
     rho_i_samples) = gibbs_sampler(x, 
                                    mu_priors, 
                                    rho_priors, 
                                    alpha_priors, 
                                    beta_priors,
                                    S)
    var_i_samples = 1 / rho_i_samples
    import pdb;pdb.set_trace()
