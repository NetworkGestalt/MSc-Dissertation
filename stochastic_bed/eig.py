"""Sequential Prior Contrastive Estimation (sPCE) lower bound on EIG."""

import math

import torch
from torch import Tensor

from .simulator import LocationFinding


def spce_bound(
    simulator: LocationFinding,
    theta: Tensor,  # [B, K*p]
    designs: Tensor,  # [B, T, D, p]
    outcomes: Tensor,  # [B, T, D]
    theta_c: Tensor | None = None,  # [B, L, K*p]
    L: int = 100,
) -> Tensor:
    """Lower bound tightens toward the true EIG as L increases (Foster et al. 2021)."""
    if theta_c is None:
        theta_c = simulator.prior().sample((theta.shape[0], L)).to(theta)  # [B, L, K*p]
    L = theta_c.shape[1]  # in case a differently-sized theta_c was passed in

    # theta_0 trajectory log-likelihood, summed over T
    logp_0 = simulator.likelihood(theta.unsqueeze(1), designs).log_prob(outcomes).sum(dim=-1)  # [B]

    # Contrastive log-likelihoods, summed over T
    lik_c = simulator.likelihood(theta_c.unsqueeze(1), designs.unsqueeze(2))
    logp_c = lik_c.log_prob(outcomes.unsqueeze(2)).sum(dim=1)  # [B, L]

    log_marginal = torch.logaddexp(logp_0, torch.logsumexp(logp_c, dim=-1)) - math.log(L + 1)  # [B]
    return logp_0 - log_marginal
