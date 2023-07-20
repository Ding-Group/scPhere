# ==============================================================================
"""The wrapped norm distribution on hyperbolic space. PyTorch version!"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class HyperbolicWrappedNorm(torch.distributions.Distribution):
    """The hyperbolic wrapped normal distribution with location `loc`
    and scale `scale`.

    Args:
        loc: Floating point tensor; the mean of the distribution(s).
        scale: Floating point tensor; the concentration of the distribution(s).
            Must contain only non-negative values.
    Raises:
        ValueError: if 'scale < 0'
    """

    arg_constraints = {'scale': torch.distributions.constraints.greater_than_eq(0)} 

    def __init__(self, loc, scale, device="cpu", dtype=torch.float32, validate_args=None):
        if validate_args:
            torch.testing.assert_close(torch.norm(loc, p=2, dim=-1), 1, atol=1e-5)
            torch._assert(self.loc.dtype == self.scale.dtype, "not of same float dtype")
        
        self.loc = loc 
        self.scale = scale
        self.base_dist = torch.distributions.normal.Normal(loc=torch.zeros_like(self.scale), scale=self.scale)
        self.dim = self.loc.shape[1] - 1
        self.device = device
        self.dtype = dtype

        super().__init__(validate_args=validate_args)

    def rsample(self, n=()):
        shape = torch.concat((torch.tensor(n), torch.tensor(self.scale.shape)), 0)
        zn = torch.randn(shape.tolist())
        zn = zn * self.scale

        shape1 = (n[0], self.batch_shape()[0], 1)
        z0 = torch.concat((torch.zeros(shape1), zn), dim=-1)

        loc0 = self.lorentzian_orig(shape1, shape.tolist())
        tmp = torch.unsqueeze(self.loc, 0)

        shape2 = (n[0], 1, 1)
        zt = self.parallel_transport(z0, loc0, torch.tile(tmp, shape2))
        z = self.exp_map(zt, torch.tile(tmp, shape2))

        return z
    
    def batch_shape(self):
        return torch.broadcast_shapes(self.loc.shape, self.loc.shape)
    
    @staticmethod
    def lorentzian_orig(s1, s0):
        x1 = torch.ones(s1)
        x0 = torch.zeros(s0)

        x_orig = torch.concat((x1, x0), dim=-1)
        return x_orig
    
    @staticmethod
    def clip_min_value(x, eps=1e-6):
        return torch.nn.functional.relu(x - eps) + eps

    def exp_map(self, x, mu):
        res = self.lorentzian_product(x, x)
        res = torch.sqrt(self.clip_min_value(res))

        res = torch.clip(res, min=0, max=32)

        return (torch.cosh(res) * mu) + (torch.sinh(res) * x / res)

    @staticmethod
    def lorentzian_product(x, y):
        y0, y1 = torch.split(y, [1, y.shape[-1] - 1], dim=-1)
        y_neg_first = torch.concat((-y0, y1), dim=-1)

        return torch.sum(torch.mul(x, y_neg_first), dim=-1, keepdim=True)

    def parallel_transport(self, x, m1, m2):
        alpha = -self.lorentzian_product(m1, m2)
        coef = self.lorentzian_product(m2, x) / (alpha + 1.0)

        return x + coef * (m1 + m2)

    def lorentz_distance(self, x, y):
        res = -self.lorentzian_product(x, y)
        res = self.clip_min_value(res, 1+1e-6)

        z = torch.sqrt(res + 1) * torch.sqrt(res - 1)

        return torch.log(res + z)

    def inv_exp_map(self, x, mu):
        alpha = -self.lorentzian_product(x, mu)
        alpha = self.clip_min_value(alpha, 1+1e-6)

        tmp = self.lorentz_distance(x, mu) / \
            torch.sqrt(alpha + 1) / torch.sqrt(alpha - 1)
        
        return tmp * (x - alpha * mu)

    def log_prob(self, x):
        v = self.inv_exp_map(x, self.loc)
        tmp = self.lorentzian_product(v, v)
        x_norm = torch.sqrt(self.clip_min_value(tmp))

        x_norm = torch.clip(x_norm, min=0, max=32)
        res = (torch.tensor(self.dim).to(dtype=torch.float32) - 1.0) * \
            torch.log(torch.sinh(x_norm) / x_norm)
        
        shape = list(self.scale.shape)
        shape1 = [self.batch_shape()[0], 1]

        loc0 = self.lorentzian_orig(shape1, shape)
        v1 = self.parallel_transport(v, self.loc, loc0)
        xx = v1[..., 1:]
        log_base_prob = torch.sum(self.base_dist.log_prob(xx), dim=-1)
        log_base_prob = torch.reshape(log_base_prob, shape1)

        return log_base_prob - res

    def prob(self, x):
        return torch.exp(self.log_prob(x))




