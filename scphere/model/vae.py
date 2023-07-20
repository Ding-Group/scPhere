import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from scphere.distributions.hyperbolic_wrapped_norm import HyperbolicWrappedNorm
from scphere.distributions.hyperspherical_uniform import HypersphericalUniform
from scphere.distributions.von_mises_fisher import VonMisesFisher
from scphere.util.util import log_likelihood_nb, log_likelihood_student

EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10

# ==============================================================================
class SCPHERE(nn.Module):
    """ Builds the scPhere model:

    Parameters
    ----------
    x:
        The N x D (cell by gene) matrix.
    n_gene: 
        The number of genes
    n_batch: 
        The number of batches for each component of the batch vector.
        In demo, we set it to 0 as there is no need to correct for batch effects. 
    z_dim: 
        The number of latent dimensions, setting to 2 for visualizations
    latent_dist: 
        One of:
            - 'vmf' for spherical latent spaces,
            - 'wn' for hyperbolic latent spaces,
            - 'normal' for normal latent spaces
    observation_dist:
        The gene expression distribution. One of:
            - 'nb' (default) for negative binomial, 
            - 'student' for student distribution
    batch_invariant: 
        Set batch_invariant=True to train batch-invariant scPhere.

        To train batch-invariant scPhere, i.e., a scPhere model taking gene expression 
        vectors only as inputs and embedding them to a latent space. 
         
        The trained model can be used to map new data, e.g., from new patients that have 
        not been seen during training scPhere (assuming patient is the major batch vector)
    activation:
        The non-linear activation function used between hidden layers in the encoder and decoder.
        Set to ELU() as default. 
    observation_dispersion:
        Dispersion rate of the observation distribution. One of:
            - 'gene' (default): NB dispersion rate is constant per gene, across all cells.
            - 'gene_batch': differnet dispersion rate per batch
    """

    def __init__(self, 
                n_gene, 
                n_batch=None, 
                z_dim=2,
                encoder_layer=None, 
                decoder_layer=None,
                latent_dist='vmf', 
                observation_dist='nb',
                batch_invariant=False, 
                activation=nn.ELU(),
                observation_dispersion='gene'
                ):
        super().__init__()
        self.n_input_feature = n_gene
        self.z_dim = z_dim
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.latent_dist = latent_dist
        self.observation_dist = observation_dist
        self.batch_invariant = batch_invariant
        self.activation = activation
        self.observation_dispersion = observation_dispersion

        if self.latent_dist == 'vmf':
            self.z_dim += 1
        
        if type(n_batch) != list:
            n_batch = [n_batch]
        self.n_batch = n_batch

        total_num_of_batches = torch.sum(torch.tensor(self.n_batch))
        # if not self.batch_invariant:
        #     self.n_input_feature = self.n_input_feature + total_num_of_batches

        ## Building Encoder
        if self.encoder_layer is None:
            self.encoder_layer = [128, 64, 32]

        layers = []
        self.encoder_layer = [self.n_input_feature + total_num_of_batches if not self.batch_invariant else self.n_input_feature] \
                            + self.encoder_layer
        for in_size, out_size in zip(self.encoder_layer[:-1], self.encoder_layer[1:]):
            lin = nn.Linear(in_features=in_size, out_features=out_size)
            bn = nn.BatchNorm1d(num_features=out_size, eps=0.001, momentum=0.99)
            layers.append(lin)
            layers.append(self.activation)
            layers.append(bn)
    
        self.encoder = nn.Sequential(*layers)

        ## Latent Distribution Layers
        if self.latent_dist == 'normal':
            self.z_mu_layer = nn.Linear(self.encoder_layer[-1], self.z_dim)
            self.z_sigma_square_layer = nn.Sequential(
                                                nn.Linear(self.encoder_layer[-1], self.z_dim), 
                                                nn.Softplus()
                                                )
        elif self.latent_dist == 'vmf':
            self.z_mu_layer = nn.Linear(self.encoder_layer[-1], self.z_dim)
            self.z_sigma_square_layer = nn.Sequential(
                                                nn.Linear(self.encoder_layer[-1], 1),
                                                nn.Softplus()
                                                )
        elif self.latent_dist == 'wn':
            self.z_mu_layer = nn.Linear(self.encoder_layer[-1], self.z_dim)
            self.z_sigma_square_layer = nn.Sequential(
                                                nn.Linear(self.encoder_layer[-1], self.z_dim),
                                                nn.Softplus()
                                                )
        else:
            raise NotImplemented

        ## Building Decoder
        if self.decoder_layer is None:
            self.decoder_layer = [32, 128]

        layers = []
        if self.latent_dist == 'wn':
            self.decoder_layer = [self.z_dim + total_num_of_batches + 1] + self.decoder_layer
        else:
            self.decoder_layer = [self.z_dim + total_num_of_batches] + self.decoder_layer

        for in_size, out_size in zip(self.decoder_layer[:-1], self.decoder_layer[1:]):
            lin = nn.Linear(in_features=in_size, out_features=out_size)
            bn = nn.BatchNorm1d(num_features=out_size, eps=0.001, momentum=0.99)
            layers.append(lin)
            layers.append(self.activation)
            layers.append(bn)
    
        self.decoder = nn.Sequential(*layers)

        ## Observation Distribution Layers
        if self.observation_dispersion == 'gene':
            self.dispersion = torch.nn.Parameter(torch.randn(self.n_input_feature))
        elif self.observation_dispersion == 'gene_batch':
            self.dispersion = torch.nn.Parameter(torch.randn(self.n_input_feature, total_num_of_batches))
        else:
            raise NotImplemented

        if self.observation_dist == 'nb':
            self.mu_layer = nn.Sequential(
                                    nn.Linear(self.decoder_layer[-1], self.n_input_feature),
                                    nn.Softmax(dim=-1)
                                    )
        else:
            self.mu_layer = nn.Linear(self.decoder_layer[-1], self.n_input_feature)
            # self.sigma_square_layer = nn.Sequential(
            #                                 nn.Linear(self.decoder_layer[-1], self.n_input_feature),
            #                                 nn.Softplus()
            #                                 )
        print("-------- Built SCPHERE model --------")

    def forward(self, x, batch):
        self.library_size = torch.sum(x, dim=1, keepdims=True)

        if len(self.n_batch) > 1:
            self.batch = self.multi_one_hot(batch, self.n_batch)
        else:
            self.batch = nn.functional.one_hot(batch[:,0], self.n_batch[0])

        self.z_mu, self.z_sigma_square = self.encode(x, self.batch)
        self.z = self.sample_from_latent_dist()
        self.mu, self.sigma_square = self.decode(self.z, self.batch)

    def encode(self, x, batch):
        if self.observation_dist == 'nb':
            x = torch.log1p(x)
            if self.latent_dist == 'vmf':
                x = nn.functional.normalize(x, p=2, dim=-1)

        if not self.batch_invariant:
            x = torch.concat((x, batch), dim=1)

        h = self.encoder(x)
        if self.latent_dist == 'normal':
            z_mu = self.z_mu_layer(h)

            z_sigma_square = self.z_sigma_square_layer(h)
        elif self.latent_dist == 'vmf':
            z_mu = self.z_mu_layer(h)
            z_mu = nn.functional.normalize(z_mu, p=2, dim=-1)

            z_sigma_square = self.z_sigma_square_layer(h) + 1
            z_sigma_square = torch.clip(z_sigma_square, 1, 10000)
        elif self.latent_dist == 'wn':
            z_mu = self.z_mu_layer(h)
            z_mu = self.polar_project(z_mu)

            z_sigma_square = self.z_sigma_square_layer(h)
        return z_mu, z_sigma_square

    def decode(self, z, batch):
        z = torch.concat((z, batch), dim=1)
        h = self.decoder(z)

        if self.observation_dist == 'nb':
            mu = self.mu_layer(h) * self.library_size

            if self.observation_dispersion == 'gene':
                sigma_square = self.dispersion
            elif self.observation_dispersion == 'gene_batch':
                sigma_square = F.linear(self.batch.type(torch.float32), self.dispersion.type(torch.float32))
            sigma_square = torch.exp(sigma_square)
        else:
            mu = self.mu_layer(h)
            sigma_square = F.linear(self.batch.type(torch.float32), 
                                    self.dispersion.type(torch.float32))
            sigma_square = torch.exp(sigma_square)
        sigma_square = torch.clip(sigma_square, EPS, MAX_SIGMA_SQUARE)

        return mu, sigma_square
        
    def sample_from_latent_dist(self):
        if self.latent_dist == 'normal':
            self.q_z = torch.distributions.normal.Normal(self.z_mu, self.z_sigma_square)
        elif self.latent_dist == 'vmf':
            self.q_z = VonMisesFisher(self.z_mu, self.z_sigma_square)
        elif self.latent_dist == 'wn':
            self.q_z = HyperbolicWrappedNorm(self.z_mu, self.z_sigma_square)
        else:
            raise NotImplemented

        return self.q_z.rsample((1,))[0,:]

    def depth_regularizer(self, x, batch):
        poisson_dist = torch.distributions.poisson.Poisson(x * 0.2)
        samples = poisson_dist.sample((1,))
        samples = torch.reshape(samples, list(x.shape))
        z_mu1, z_sigma_square1 = self.encode(nn.functional.relu(x - samples), batch)

        mean_diff = torch.sum(torch.pow(self.z_mu - z_mu1, 2), dim=1)
        loss = torch.mean(mean_diff)

        return loss
    
    def ELBO(self, x):
        ## Getting the log-likelihood
        if self.observation_dist == 'student':
            self.log_likelihood = torch.mean(
                log_likelihood_student(x,
                                    self.mu,
                                    self.sigma_square,
                                    df=5.0))
        elif self.observation_dist == 'nb':
            self.log_likelihood = torch.mean(
                log_likelihood_nb(x,
                                self.mu,
                                self.sigma_square,
                                eps=1e-10))

        ## Getting KL divergence
        if self.latent_dist == 'normal':
            self.p_z = torch.distributions.normal.Normal(torch.zeros_like(self.z),
                                                         torch.ones_like(self.z))

            kl = torch.distributions.kl_divergence(self.q_z, self.p_z)
            self.kl = torch.mean(torch.sum(kl, dim=-1))
        elif self.latent_dist == 'vmf':
            self.p_z = HypersphericalUniform(dim=self.z_dim - 1, validate_args=None)

            kl = torch.distributions.kl_divergence(self.q_z, self.p_z)
            self.kl = torch.mean(kl)
        elif self.latent_dist == 'wn':
            tmp = self.polar_project(torch.zeros_like(self.z_sigma_square))
            self.p_z = HyperbolicWrappedNorm(tmp,
                                                    torch.ones_like(self.z_sigma_square))

            kl = self.q_z.log_prob(self.z) - self.p_z.log_prob(self.z)
            self.kl = torch.mean(kl)
        else:
            raise NotImplemented
        
        return self.log_likelihood - self.kl
    
    def polar_project(self, x):
        x_norm = torch.sum(torch.square(x), dim=1, keepdims=True)
        x_norm = torch.sqrt(self.clip_min_value(x_norm))

        x_unit = x / torch.reshape(x_norm, (-1, 1))
        x_norm = torch.clip(x_norm, min=0, max=32)

        z = torch.concat((torch.cosh(x_norm), torch.sinh(x_norm) * x_unit), dim=1)

        return z

    @staticmethod
    def clip_min_value(x, eps=EPS):
        return nn.functional.relu(x - eps) + eps

    def multi_one_hot(self, indices, depth_list):
        one_hot_tensor = nn.functional.one_hot(indices[:,0], depth_list[0])
        for col in range(1, len(depth_list)):
            next_one_hot = nn.functional.one_hot(indices[:,col], depth_list[col])
            one_hot_tensor = torch.concat((one_hot_tensor, next_one_hot), dim=1)
        
        return one_hot_tensor

    def get_log_likelihood(self, x, batch):
        num_samples = 5
        log_likelihood_value = 0
        for i in range(num_samples):
            self.forward(x, batch)

            if self.observation_dist == 'nb':
                log_likelihood = log_likelihood_nb(x, self.mu, self.sigma_square)
            else:
                dof = 2.0
                log_likelihood = log_likelihood_student(x, self.mu, self.sigma_square, df=dof)

            log_likelihood_value += log_likelihood

        log_likelihood_value /= np.float32(num_samples)

        return log_likelihood_value

