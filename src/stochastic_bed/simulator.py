"""Location-finding simulator: sensors observe signals with log-Gaussian noise."""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor


class Trajectory(NamedTuple):
    designs: Tensor  # [B, T, D, p]
    outcomes: Tensor  # [B, T, D]
    entropies: Tensor | None  # [B, T], or None for policies without entropy


class LocationFinding(nn.Module):
    def __init__(
        self,
        p: int = 2,  # Number of coordinate dimensions
        K: int = 1,  # Number of sources
        D: int = 1,  # Number of sensors each time period
        T: int = 7,  # Number of time periods
        a: float = 1.0,  # Source signal weight
        b: float = 0.1,  # Background signal weight
        m: float = 0.001,  # Min squared distance to a source
        noise_std: float = 0.5,  # Observation noise std
        prior_mean: float | Tensor = 0.0,  # Gaussian prior mean, scalar or [K*p]
        prior_std: float = 1.0,  # Gaussian prior std (isotropic)
    ) -> None:
        super().__init__()

        self.p = p
        self.K = K
        self.D = D
        self.T = T
        self.a = a
        self.b = b
        self.m = m
        self.noise_std = noise_std
        self.prior_std = prior_std

        mean = torch.as_tensor(prior_mean, dtype=torch.get_default_dtype())
        self.register_buffer("prior_mean", mean.expand(self.K * self.p).clone())

    def prior(self):
        base = dist.Normal(self.prior_mean, self.prior_std)
        return dist.Independent(base, 1)  # event shape: [K*p]

    def likelihood(self, theta: Tensor, designs: Tensor):
        """Outcome likelihood p(y | theta, design) as a Distribution."""
        theta = theta.unflatten(-1, (self.K, self.p))  # [B, K, p]

        diffs = designs.unsqueeze(-2) - theta.unsqueeze(-3)  # [B, D, K, p]
        sq_distances = diffs.pow(2).sum(-1)  # [B, D, K]
        signals = self.a / (self.m + sq_distances)
        total_signal = signals.sum(dim=-1) + self.b  # [B, D]
        loc = torch.log(total_signal)

        return dist.Independent(dist.Normal(loc, self.noise_std), 1)

    def step(self, theta: Tensor, design: Tensor):
        """Sample an outcome y_t for a design given theta."""
        y_t = self.likelihood(theta, design).rsample()  # [B, D]
        return y_t

    def rollout(self, theta: Tensor, policy: nn.Module) -> Trajectory:
        """Simulate full trajectories under a given batch of thetas and policy."""
        B = theta.shape[0]
        designs, outcomes, entropies = [], [], []

        hist_designs = theta.new_zeros(B, 0, self.D, self.p)
        hist_outcomes = theta.new_zeros(B, 0, self.D)

        for _ in range(self.T):
            xi_t, entropy_t = policy(hist_designs, hist_outcomes)
            y_t = self.step(theta, xi_t)

            designs.append(xi_t)
            outcomes.append(y_t)
            entropies.append(entropy_t)
            hist_designs = torch.cat([hist_designs, xi_t.unsqueeze(1)], dim=1)
            hist_outcomes = torch.cat([hist_outcomes, y_t.unsqueeze(1)], dim=1)

        return Trajectory(
            designs=torch.stack(designs, dim=1),
            outcomes=torch.stack(outcomes, dim=1),
            entropies=torch.stack(entropies, dim=1) if entropies[0] is not None else None,
        )
