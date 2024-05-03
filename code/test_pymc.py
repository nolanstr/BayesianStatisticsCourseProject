import numpy as np
from scipy.stats import norm, lognorm
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns

def create_df(posterior, priors, key):
    
    data = []
    dist_keys = []

    _prior = priors[key].to_numpy().flatten()
    _posterior = posterior[key].to_numpy().flatten()
    _prior_dist_keys = ["Prior"]*_prior.shape[0]
    _posterior_dist_keys = ["Posterior"]*_posterior.shape[0]

    data.append(_prior)
    data.append(_posterior)
    dist_keys.append(_prior_dist_keys)
    dist_keys.append(_posterior_dist_keys)

    _dict = {"data":np.concatenate(data), 
             "Distribution":np.concatenate(dist_keys)}
    df = pd.DataFrame.from_dict(_dict)

    return df

def plot_between_group(trace, priors, tag, var, kde=True, show=True):

    df = create_df(trace.posterior, 
                   priors.prior, 
                   tag)
    fig, ax = plt.subplots()
    if kde:
        sns.kdeplot(data=df, x="data", hue="Distribution", 
                     fill=True)
    else:
        sns.histplot(data=df, x="data", hue="Distribution")
    fig.suptitle(f"Between Group {var}", fontsize=16)
    ax.set_xlabel(var, fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    plt.setp(ax.get_legend().get_texts(), fontsize='14') 
    plt.setp(ax.get_legend().get_title(), fontsize='16')
    plt.tight_layout()
    plt.savefig(tag, dpi=300)
    if show:
        plt.show()

def make_interval(mu, var, data):
    mu = np.mean(mu)
    var = np.mean(var)
    std = np.sqrt(var)
    l = min((mu - 3 * std, data.min()))
    u = max((mu + 3 * std, data.max()))
    return np.linspace(l, u, 1000)

def plot_within_group_norm(trace, data, samples=50, show=True):
    _mu = trace.posterior["within_group_means"].to_numpy()
    _var = trace.posterior["within_group_variances"].to_numpy()
    n_groups = _mu.shape[-1]
    
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12,8))
    axs = np.ravel(axs)
    axs_labels = [r"$\alpha_{1}$", r"$\alpha_{2}$", r"$\alpha_{3}$", 
                  r"$\alpha_{4}$", r"$\alpha_{5}$", r"$\alpha_{6}$", 
                  r"$\alpha_{7}$", r"$\alpha_{8}$", r"$\alpha_{9}$", 
                  r"$\alpha_{10}$", r"$\alpha_{11}$", r"$\alpha_{12}$"]  
    for i in range(n_groups):
        _mus = _mu[:,:,i].flatten()
        _vars = _var[:,:,i].flatten()
        _stds = np.sqrt(_vars)
        _data = data[i]
        X = make_interval(_mus, _vars, _data)

        for j in range(samples):
            axs[i].plot(X, 
                        norm(loc=_mus[-j], scale=_stds[-j]).pdf(X),
                        alpha=0.1,
                        color="b")
        axs[i].plot([], [], color="b", label="Posterior")
        axs[i].scatter(_data, np.zeros_like(_data), color="k", 
                       label="Data")
        axs[i].set_title(axs_labels[i])
        axs[i].legend(loc="upper right")
        axs[i].set_xlabel("Equivalent Radius")
        axs[i].set_ylabel(r"$P(Y_{i}|\mu_{i}, \sigma_{i}^{2})$")
    fig.suptitle(
    "Within-Group Posterior Distributions, $Y_{i} \sim Normal(\mu_{i}, \sigma_{i}^{2})$",
    fontsize=16)
    fig.suptitle("Posterior")
    plt.tight_layout()
    plt.savefig("within_group_normal", dpi=300)
    if show:
        plt.show()

def plot_within_group_lognorm(trace, data, samples=50, show=True):
    _mu = trace.posterior["within_group_means"].to_numpy()
    _var = trace.posterior["within_group_variances"].to_numpy()
    n_groups = _mu.shape[-1]
    
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12,8))
    axs = np.ravel(axs)
    axs_labels = [r"$\alpha_{1}$", r"$\alpha_{2}$", r"$\alpha_{3}$", 
                  r"$\alpha_{4}$", r"$\alpha_{5}$", r"$\alpha_{6}$", 
                  r"$\alpha_{7}$", r"$\alpha_{8}$", r"$\alpha_{9}$", 
                  r"$\alpha_{10}$", r"$\alpha_{11}$", r"$\alpha_{12}$"]  
    for i in range(n_groups):
        _mus = _mu[:,:,i].flatten()
        _vars = _var[:,:,i].flatten()
        _stds = np.sqrt(_vars)
        _data = data[i]
        X = make_interval(_mus, _vars, _data)

        for j in range(samples):
            weights = norm(loc=_mus[-j], scale=_stds[-j]).pdf(X)
            axs[i].plot(np.exp(X),
                        weights,
                        alpha=0.1,
                        color="b")
        axs[i].plot([], [], color="b", label="Posterior")
        axs[i].scatter(np.exp(_data), np.zeros_like(_data), color="k", 
                       label="Data")
        axs[i].set_title(axs_labels[i])
        axs[i].legend(loc="upper right")
        axs[i].set_xlabel("Equivalent Radius")
        axs[i].set_ylabel(r"$P(Y_{i}|\mu_{i}, \sigma_{i}^{2})$")
    fig.suptitle(
    "Within-Group Posterior Distributions, $Y_{i} \sim LogNormal(\mu_{i}, \sigma_{i}^{2})$",
    fontsize=16)
    plt.tight_layout()
    plt.savefig("within_group_lognormal", dpi=300)
    if show:
        plt.show()
if __name__ == "__main__":
    # Initialize data

    df = pd.read_csv("../single_scan.csv")
    data = []
    variant_ids = np.unique(df["alpha_variant"])
    for variant_id in variant_ids:
        variant_df = df.loc[df["alpha_variant"] == variant_id]
        equivalent_diameter = variant_df["equivalent_diameter"].to_numpy()
        equivalent_radius = equivalent_diameter / 2
        data.append(np.log(equivalent_radius))  # log data to make normal

    num_groups = len(data)

    # True parameters
    true_between_group_mean = 10
    true_between_group_variance = 2
    true_between_group_alpha = 2
    true_between_group_beta = 2
    true_within_group_alpha = 2
    true_within_group_beta = 2

    # Model
    with pm.Model() as hierarchical_model:
        # Hyperparameters
        between_group_mean = pm.Normal("between_group_mean", mu=0, sigma=10)
        between_group_variance = pm.InverseGamma(
            "between_group_variance", alpha=2, beta=2
        )
        between_group_alpha = pm.Geometric("between_group_alpha", 
                                           p=0.5)
        between_group_beta = pm.InverseGamma("between_group_beta", 
                                             alpha=2, 
                                             beta=2)

        # Individual group parameters
        within_group_means = pm.Normal(
            "within_group_means",
            mu=between_group_mean,
            sigma=np.sqrt(between_group_variance),
            shape=num_groups,
        )
        within_group_variances = pm.InverseGamma(
            "within_group_variances",
            alpha=between_group_alpha,
            beta=between_group_beta,
            shape=num_groups,
        )

        # Likelihood
        for i in range(num_groups):
            pm.Normal(
                f"group_{i}_data",
                mu=within_group_means[i],
                sigma=np.sqrt(within_group_variances[i]),
                observed=data[i],
            )

        # Sample from Hierarchical Model
        trace = pm.sample(
            2000, tune=1000, cores=2
        )  
        priors = pm.sample_prior_predictive(samples=10000)

        # Plotting prior vs posterior for between-group Variables 

        #plot_between_group(trace, priors, "between_group_mean", r"$\mu$")
        #plot_between_group(trace, priors, 
        #                   "between_group_variance", r"$\sigma^{2}$")
        #plot_between_group(trace, priors, "between_group_alpha", 
        #                   r"$\alpha$",
        #                   kde=False)
        #plot_between_group(trace, priors, "between_group_beta", 
        #                   r"$\beta$")
        #plot_within_group_norm(trace, data, show=True)
        plot_within_group_lognorm(trace, data, show=True)
        
        # Plotting Posterior distributions vs. Samples

        import pdb;pdb.set_trace()
        
