import numpy as np
import pandas as pd

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
    
    sample_means = np.array([x_.mean() for x_ in x])
    sample_vars = np.array([x_.var() for x_ in x])

    mean_of_means = np.mean(sample_means)
    var_of_means = np.var(sample_means)

    mean_of_vars = np.mean(sample_vars)
    var_of_vars = np.var(sample_vars)

    import pdb;pdb.set_trace()
