import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import colorcet as cc
from matplotlib import pyplot as plt
import EqualEarth


data_dir = './example/data/'
result_dir = './example/demo-out/datasets/epi_cells/'
types = pd.read_csv(data_dir + 'uc_epi_celltype.tsv', sep='\t', header=None).to_numpy()
z_mu = pd.read_csv(result_dir + 'vmf_elu_epi_latent.tsv', sep=' ', header=None).to_numpy()

x = z_mu[:,0]
y = z_mu[:,1]
z = z_mu[:,2]

lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2 )))
longs = np.degrees(np.arctan2(y, x))

# 2D Equal Earth projection
mpl.rcParams['figure.dpi']= 500
palette = sns.color_palette(cc.glasbey_dark, n_colors=len(np.unique(types)))

fig = plt.figure('Equal Earth Projection', figsize=(15, 8))
ax = fig.add_subplot(111, projection='equal_earth')
sns.scatterplot(x=longs, y=lat, data=z_mu, hue=types[:,0], palette=palette, s=5, legend=False)

plt.xlabel('latent dimension 1')
plt.ylabel('latent dimension 2')
plt.savefig(result_dir + 'vmf_elu_epi_latents.png')