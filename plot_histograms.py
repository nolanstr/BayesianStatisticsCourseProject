import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Initialize data

df = pd.read_csv("./single_scan.csv")
x = []
variant_ids = np.unique(df["alpha_variant"])
for variant_id in variant_ids:
    variant_df = df.loc[df["alpha_variant"] == variant_id]
    equivalent_diameter = variant_df["equivalent_diameter"].to_numpy()
    equivalent_radius = equivalent_diameter / 2
    x.append(equivalent_radius)
 
# Create a 4x3 grid of histogram plots
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 8))

# Flatten the axes array to iterate over all subplots
axes = axes.flatten()

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
# Plot histograms for each feature
for i in range(len(x)):
    ax = axes[i]
    ax.hist(x[i], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(labels[i])
    ax.set_xlabel('Equivalent Radius')
    ax.set_ylabel('Frequency')
    ax.grid(True)

# Hide empty subplots if there are fewer features than subplots
for i in range(len(x), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("equivalent_radius_hist", dpi=300)
plt.show()
