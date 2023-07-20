import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import colorcet as cc
from matplotlib import pyplot as plt
from scphere.util.util import read_mtx
import EqualEarth


def plot_latents(save_path, embeddings, cell_types, latent_dist):
    mpl.rcParams['figure.dpi']= 400

    if cell_types is not None:
        palette = sns.color_palette(cc.glasbey_dark, n_colors=len(np.unique(cell_types)))
    
    if latent_dist == 'vmf':
        # equal earth projection
        x = embeddings[:,0]
        y = embeddings[:,1]
        z = embeddings[:,2]

        lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2 )))
        longs = np.degrees(np.arctan2(y, x))

        # plotting
        fig = plt.figure('Equal Earth Projection', figsize=(12, 6))
        ax = fig.add_subplot(111, projection='equal_earth')

        if cell_types is not None:
            sns.scatterplot(x=longs, y=lat, data=embeddings, hue=cell_types[:,0], palette=palette, s=5, legend=False)
        else:
            sns.scatterplot(x=longs, y=lat, data=embeddings, s=5, legend=False)
    elif latent_dist == 'wn':
        # Embeddings are 3d, changing them to 2d
        new_embeddings = np.zeros((embeddings.shape[0], 2))
        new_embeddings[:,0] = embeddings[:,1]/(1+embeddings[:,0])
        new_embeddings[:,1] = embeddings[:,2]/(1+embeddings[:,0])

        sns.set(rc={'figure.figsize':(12,12)})
        if cell_types is not None:
            fig, ax = plt.subplots(1,1)
            ax = sns.scatterplot(x=new_embeddings[:,0], y=new_embeddings[:,1], data=new_embeddings, hue=cell_types[:,0], palette=palette)
        else:
            sns.scatterplot(x=new_embeddings[:,0], y=new_embeddings[:,1], data=new_embeddings, legend=False)
    elif latent_dist == 'normal':
        sns.set(rc={'figure.figsize':(12,8)})
        if cell_types is not None:
            fig, ax = plt.subplots(1,1)
            ax = sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], data=embeddings, hue=cell_types[:,0], palette=palette)
        else:
            sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], data=embeddings, legend=False)

    plt.xlabel('latent dimension 1')
    plt.ylabel('latent dimension 2')
    plt.savefig(save_path + '_latents.png')