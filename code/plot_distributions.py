import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import lognorm



labels = [ r"$\alpha_{1}$", 
          r"$\alpha_{2}$", 
          r"$\alpha_{3}$", 
          r"$\alpha_{4}$", 
          r"$\alpha_{5}$", 
          r"$\alpha_{6}$", 
          r"$\alpha_{7}$", 
          r"$\alpha_{8}$", 
          r"$\alpha_{9}$", 
          r"$\alpha_{10}$", 
          r"$\alpha_{11}$", 
          r"$\alpha_{12}$"]

def visualize_lognormal_distributions(data, mu_samples, var_samples,
                                      n_samples=25):

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 8))
    axes = axes.flatten()

    for i in range(len(data)):

        ax = axes[i]

        mu_i_samples = mu_samples[-n_samples:,i]
        var_i_samples = var_samples[-n_samples:,i]
        x_i = data[i]
        upper = x_i.max() * 1.001
        X = np.linspace(0, upper, 1000)
        ax.scatter(x_i, np.zeros_like(x_i), color="k")
        
        for mu, var in zip(mu_i_samples, var_i_samples):
            d = lognorm(s=np.sqrt(var), scale=np.exp(mu))
            print(f"Expectation = {d.mean()}")
            ax.plot(X, d.pdf(X), color='skyblue')
        ax.set_title(labels[i])
        ax.set_xlabel('Equivalent Radius')
        ax.set_ylabel(r'$Pr(Y_{i}|\mu, \sigma^{2})$')

    for i in range(len(data), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("data_lognormal_dist", dpi=300)
    plt.show()

def plot_prior_vs_posterior(model, label, tag, burnin=50, show=False):
    filename = f"prior_vs_post/{tag}"
    fig, ax = plt.subplots()
    if "rho" in tag:
        interval = model._distribution.interval(0.95)
        X = np.linspace(interval[0], interval[1], 1000)
        weights = model._distribution_df(X)
        sns.kdeplot(x=X, weights=weights, ax=ax, label="Prior", fill=True,
                    cut=0)
        sns.kdeplot(x=1/model.samples[burnin:], 
                    ax=ax, 
                    label="Posterior", 
                    fill=True,
                    cut=0)
    elif "alpha" in tag:
        X = np.linspace(0, 20, 1000)
        weights = model._distribution_df(X)
        sns.kdeplot(x=X, weights=weights, ax=ax, label="Prior", fill=True,
                    cut=0)
        #sns.kdeplot(x=model.samples[burnin:], 
        #            ax=ax, 
        #            label="Posterior", 
        #            fill=True,
        #            cut=0)
        ax.axvline(model.samples[-1], label="Posterior", color="orange")
    else:
        interval = model._distribution.interval(0.95)
        X = np.linspace(interval[0], interval[1], 1000)
        weights = model._distribution_df(X)
        sns.kdeplot(x=X, weights=weights, ax=ax, label="Prior", fill=True,
                    cut=0)
        sns.kdeplot(x=model.samples[burnin:], 
                    ax=ax, 
                    label="Posterior", 
                    fill=True,
                    cut=0)
    ax.legend()
    ax.set_xlabel(label)
    fig.suptitle(f"Posterior vs. Prior for {label}")
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.clf()
    return
