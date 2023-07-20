import numpy as np
import scipy.sparse as sparse
import torch
import logging

from scphere.distributions.von_mises_fisher import VonMisesFisher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
def log_likelihood_student(x, mu, sigma_square, df=2.0):
    sigma = torch.sqrt(sigma_square)
    dist = torch.distributions.studentT.StudentT(df=df, loc=mu, scale=sigma)

    return torch.sum(dist.log_prob(x), dim=1)

def log_likelihood_vmf(x, mu, sigma_square):
    dist = VonMisesFisher(loc=mu, scale=sigma_square)
    return torch.sum(dist.log_prob(x), dim=1)

def log_likelihood_nb(x, mu, sigma, eps=1e-16):
    log_mu_sigma = torch.log(mu + sigma + eps)

    ll = torch.lgamma(x + sigma) - torch.lgamma(sigma) - \
        torch.lgamma(x + 1) + sigma * torch.log(sigma + eps) - \
        sigma * log_mu_sigma + x * torch.log(mu + eps) - x * log_mu_sigma

    return torch.sum(ll, dim=-1)

def transform_mtx(x, transform_type='log'):
    if type(x) is not sparse.coo_matrix:
        x = sparse.coo_matrix(x)

    if transform_type == 'log':
        x.data = np.log(x.data + 1)
    elif transform_type == 'log2':
        x.data = np.log2(x.data + 1)
    elif transform_type == 'log10':
        x.data = np.log10(x.data + 1)
    else:
        logger.error(f'transform_type should be either log, log2, or log10')

    return x


def read_mtx(filename, dtype='int32'):
    from scipy.io import mmread
    x = mmread(filename).astype(dtype)

    return x
