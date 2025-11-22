"""
Noise schedules for discrete diffusion.
Taken from: mdlm/noise_schedule.py

Only includes LogLinearNoise as recommended in IMPLEMENTATION_ANALYSIS.md
"""

import torch
import torch.nn as nn


class LogLinearNoise(nn.Module):
    """Log Linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def forward(self, t):
        """
        Baseline forward method to get the total + rate of noise at a timestep.

        Args:
            t: timestep tensor

        Returns:
            (total_noise, rate_noise) tuple
        """
        return self.total_noise(t), self.rate_noise(t)

    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t).

        Args:
            t: timestep tensor

        Returns:
            rate of noise
        """
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0).

        Args:
            t: timestep tensor

        Returns:
            total noise accumulated
        """
        return -torch.log1p(-(1 - self.eps) * t)

    def importance_sampling_transformation(self, t):
        """
        Importance sampling transformation for variance reduction.

        Args:
            t: timestep tensor

        Returns:
            transformed timestep
        """
        f_T = torch.log1p(- torch.exp(- self.sigma_max))
        f_0 = torch.log1p(- torch.exp(- self.sigma_min))
        sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
        t = - torch.expm1(- sigma_t) / (1 - self.eps)
        return t
