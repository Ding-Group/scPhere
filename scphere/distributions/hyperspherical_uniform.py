# ==============================================================================
"""Hyperspherical Uniform distribution modified from the hyperspherical_vae package ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from torch.distributions import Distribution, constraints

class HypersphericalUniform(Distribution):
    """Hyperspherical Uniform distribution with `dim` parameter.

    Args:
        dim: Integer tensor, dimensionality of the distribution(s). Must
            be `dim > 0`.
    Raises:
        ValueError: if 'dim <= 0'

    #### Mathematical Details

    """

    arg_constraints={}

    def __init__(self, dim, device="cpu", dtype=torch.float32, validate_args=None):
        self.dim = dim
        self.device = device
        self.dtype = dtype

        super().__init__(validate_args=validate_args)
        
    def rsample(self, n=()):
        sample = torch.empty(size= n + (self.dim,), device=self.device, dtype=self.dtype)
        sample = torch.nn.init.normal_(sample)

        return torch.nn.functional.normalize(sample, dim=-1)

    def log_prob(self, x):
        return torch.full_like(x[..., 0], 
            self.log_surface_area(), device=self.device, dtype=self.dtype)

    def log_surface_area(self):
        return -math.lgamma((self.dim + 1) / 2) + \
            math.log(2) + ((self.dim + 1) / 2) * math.log(math.pi)

    def entropy(self):
        return self.log_surface_area()

    def prob(self, x):
        return torch.exp(self.log_prob(x))



            