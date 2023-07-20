import numpy as np
import torch
from torch import nn
import pandas as pd
from matplotlib import pyplot as plt
from scphere.util.util import read_mtx
from scphere.util.trainer import Trainer
from scphere.model.vae import SCPHERE
from scphere.util.plot import plot_trace
from example.script.plot_latents import plot_latents

## Getting epi data
data_path = './example/data/'
save_path = './example/demo-out/vmf_50epoch_epi'

mtx = data_path + 'uc_epi.mtx'
x = read_mtx(mtx)
x = x.transpose().todense()
# Getting the cell types to plot them with the embeddings, set to None if not available
cell_types = pd.read_csv(data_path + 'uc_epi_celltype.tsv', sep='\t', header=None).to_numpy()

## Batch vector(s)
# This should be changed if using a dataset that has more than one batch files (e.g. patient, health, location).
# If that is the case, the batch vectors should be concatenated into a batch matrix, for example if we import:
#              batch_p = the batch vector of patient
#              batch_h = the batch vector of health 
# Then the overall batch matrix is:
#              batch = pd.concat([batch_p.iloc[:, 0], batch_h.iloc[:, 0]], axis=1).to_numpy()
# And the n_batch array is:
#              n_batch_p = len(np.unique(batch_p))
#              n_batch_h = len(np.unique(batch_h))
#              n_batch = [n_batch_p, n_batch_h]
# If there's no batches, you can set the batch vector to -1 and n_batch=0:
#              batch = np.zeros(x.shape[0]) * -1

batch_h = pd.read_csv(data_path + 'uc_epi_batch_health.tsv', sep='\t', header=None)
batch_l = pd.read_csv(data_path + 'uc_epi_batch_location.tsv', sep='\t', header=None)
batch_p = pd.read_csv(data_path + 'uc_epi_batch_patient.tsv', sep='\t', header=None)
batch = pd.concat([batch_h.iloc[:,0], batch_l.iloc[:,0], batch_p.iloc[:,0]], axis=1).to_numpy()
n_batch = [len(np.unique(batch_h)), len(np.unique(batch_l)), len(np.unique(batch_p))]

## Building the model
model = SCPHERE(n_gene=x.shape[1], n_batch=n_batch, batch_invariant=False, z_dim=2, latent_dist='vmf', observation_dispersion='gene')
model.train()

## Training the model
trainer = Trainer(model=model, x=x, batch_id=batch, max_epoch=50, mb_size=128, learning_rate=0.001)
trainer.train()

## Saving the trained model
torch.save(model.state_dict(), save_path + '_model.pth')

## Embedding all the data and saving the posterior means
model.eval()
model.forward(torch.tensor(x), torch.tensor(batch))
z_mu = model.z_mu.detach().numpy()
np.savetxt(save_path + '_latent.tsv', z_mu)

## Plotting log-likelihood and kl-divergence at each iteration
plot_trace([np.arange(len(trainer.status['kl_divergence']))*50] * 2,
            [trainer.status['log_likelihood'], trainer.status['kl_divergence']],
            ['log_likelihood', 'kl_divergence'])
plt.savefig(save_path + '_train.png')

## Plotting the latents
plot_latents(save_path, embeddings=z_mu, cell_types=cell_types, latent_dist=model.latent_dist)

 