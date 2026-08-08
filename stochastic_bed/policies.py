"""Policy networks map history to a [D, p] design (D sensors, p coordinates) and its std."""

import math

import torch
import torch.nn as nn
import torch.distributions as dist


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        dims = (input_dim, *hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = MLP(input_dim, output_dim, hidden_dims)

    def forward(self, hist_designs, hist_outcomes):
        B, T = hist_designs.shape[:2]

        if T == 0:
            return hist_designs.new_zeros(B, self.output_dim)

        xi_flat = hist_designs.flatten(start_dim=2)  # [B, T, D*p]
        x = torch.cat([xi_flat, hist_outcomes], dim=-1)  # [B, T, D*p + D]

        z = self.mlp(x)  # [B, T, output_dim]
        return z.sum(dim=1)  # [B, output_dim]


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        D: int,
        p: int,
        design_bound: float,
        enc_hidden_dims: tuple[int, ...] = (256, 256),
        enc_output_dim: int = 128,
        hidden_dims: tuple[int, ...] = (64,),
    ):
        super().__init__()
        self.D = D
        self.p = p
        self.design_bound = design_bound

        self.history_encoder = HistoryEncoder(
            input_dim=D * p + D, output_dim=enc_output_dim, hidden_dims=enc_hidden_dims
        )

        self.mlp = MLP(enc_output_dim, D * p, hidden_dims)

    def forward(self, hist_designs, hist_outcomes):
        enc = self.history_encoder(hist_designs, hist_outcomes)  # [B, enc_output_dim]
        z = self.mlp(enc).unflatten(-1, (self.D, self.p))  # [B, D, p]
        design = torch.tanh(z) * self.design_bound
        return design, None


class StochasticPolicy(nn.Module):
    def __init__(
        self,
        D: int,
        p: int,
        design_bound: float,
        enc_hidden_dims: tuple[int, ...] = (256, 256),
        enc_output_dim: int = 128,
        hidden_dims: tuple[int, ...] = (64,),
        min_std: float = 0.01,
        init_mean: float = 0.0,
        init_std: float = 0.5,
    ):
        super().__init__()
        self.D = D
        self.p = p
        self.design_bound = design_bound
        self.min_std = min_std

        self.history_encoder = HistoryEncoder(
            input_dim=D * p + D,
            output_dim=enc_output_dim,
            hidden_dims=enc_hidden_dims,
        )

        self.mean_mlp = MLP(enc_output_dim, D * p, hidden_dims)
        nn.init.constant_(self.mean_mlp.net[-1].bias, init_mean)

        self.log_std_mlp = MLP(enc_output_dim, D * p, hidden_dims)
        nn.init.constant_(self.log_std_mlp.net[-1].bias, math.log(max(init_std, 1e-8)))

    def _base_distribution(self, hist_designs, hist_outcomes):
        enc = self.history_encoder(hist_designs, hist_outcomes)  # [B, enc_output_dim]

        mean = self.mean_mlp(enc).unflatten(-1, (self.D, self.p))  # [B, D, p]
        log_std = self.log_std_mlp(enc).unflatten(-1, (self.D, self.p))  # [B, D, p]
        std = torch.exp(log_std) + self.min_std

        return dist.Independent(
            dist.Normal(mean, std),
            reinterpreted_batch_ndims=2,
        )  # Event shape: [D, p]

    def forward(self, hist_designs, hist_outcomes):
        base_dist = self._base_distribution(hist_designs, hist_outcomes)
        design = torch.tanh(base_dist.rsample()) * self.design_bound  # [B, D, p]
        return design, base_dist.base_dist.stddev  # [B, D, p]


class StaticDeterministicPolicy(nn.Module):
    def __init__(self, D: int, p: int, design_bound: float, T: int, init_scale: float = 0.5):
        super().__init__()
        self.D = D
        self.p = p
        self.design_bound = design_bound
        self.z = nn.Parameter(init_scale * torch.randn(T, D, p))

    def forward(self, hist_designs, hist_outcomes):
        B, t = hist_designs.shape[:2]
        design = torch.tanh(self.z[t]) * self.design_bound  # [D, p]
        return design.unsqueeze(0).expand(B, -1, -1), None


class StaticStochasticPolicy(nn.Module):
    def __init__(
        self,
        D: int,
        p: int,
        design_bound: float,
        T: int,
        min_std: float = 0.01,
        init_std: float = 0.5,
        init_scale: float = 0.5,
    ):
        super().__init__()
        self.D = D
        self.p = p
        self.design_bound = design_bound
        self.min_std = min_std

        self.mean = nn.Parameter(init_scale * torch.randn(T, D, p))
        self.log_std = nn.Parameter(torch.full((T, D, p), math.log(max(init_std, 1e-8))))

    def _base_distribution(self, hist_designs, hist_outcomes):
        B, t = hist_designs.shape[:2]
        mean = self.mean[t].unsqueeze(0).expand(B, -1, -1)  # [B, D, p]
        std = (torch.exp(self.log_std[t]) + self.min_std).unsqueeze(0).expand(B, -1, -1)

        return dist.Independent(dist.Normal(mean, std), reinterpreted_batch_ndims=2)

    def forward(self, hist_designs, hist_outcomes):
        base_dist = self._base_distribution(hist_designs, hist_outcomes)
        design = torch.tanh(base_dist.rsample()) * self.design_bound  # [B, D, p]
        return design, base_dist.base_dist.stddev  # [B, D, p]


class RandomPolicy(nn.Module):
    def __init__(self, D: int, p: int, design_bound: float):
        super().__init__()
        self.D = D
        self.p = p
        self.design_bound = design_bound

    def forward(self, hist_designs, hist_outcomes):
        B = hist_designs.shape[0]
        b = self.design_bound
        design = hist_designs.new_empty(B, self.D, self.p).uniform_(-b, b)
        return design, None
