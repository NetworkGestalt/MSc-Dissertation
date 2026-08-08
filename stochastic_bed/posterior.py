"""Posterior network maps history to posterior density over theta (K*p)."""

import os

os.environ["KERAS_BACKEND"] = "torch"

import torch.nn as nn
from bayesflow.networks import CouplingFlow, FlowMatching

from .policies import HistoryEncoder


class PosteriorNet(nn.Module):
    def __init__(
        self,
        D: int,
        p: int,
        K: int,
        enc_hidden_dims: tuple[int, ...] = (256, 256),
        enc_output_dim: int = 128,
        flow_type: str = "CouplingFlow",  # or "FlowMatching"
    ):
        super().__init__()

        self.history_encoder = HistoryEncoder(
            input_dim=D * p + D,
            output_dim=enc_output_dim,
            hidden_dims=enc_hidden_dims,
        )

        if flow_type == "CouplingFlow":
            self.flow = CouplingFlow(
                subnet="mlp",
                depth=6,
                transform="affine",
                permutation="random",
                use_actnorm=True,
                base_distribution="normal",
            )

        elif flow_type == "FlowMatching":
            self.flow = FlowMatching(
                subnet="time_mlp",
                subnet_kwargs={"widths": (256,) * 6},
                base_distribution="normal",
                use_optimal_transport=True,
                loss_fn="mse",
            )

        else:
            raise ValueError(f"Unknown flow_type: {flow_type!r}")

        self.flow.build(
            xz_shape=(32, K * p),
            conditions_shape=(32, enc_output_dim),
        )  # Batch dim (32) is arbitrary

    def loss(self, theta, designs, outcomes):
        """NLL for CouplingFlow, velocity regression for FlowMatching."""
        enc = self.history_encoder(designs, outcomes)  # [B, enc_output_dim]
        return self.flow.compute_metrics(theta, conditions=enc, stage="training")["loss"]

    def log_prob(self, theta, designs, outcomes):
        enc = self.history_encoder(designs, outcomes)
        return self.flow.log_prob(theta, conditions=enc)  # [B]

    def sample(self, n_samples, designs, outcomes):
        B = designs.shape[0]
        enc = self.history_encoder(designs, outcomes)  # [B, enc_output_dim]
        enc = enc.unsqueeze(0).expand(n_samples, -1, -1)  # [n_samples, B, enc_output_dim]
        return self.flow.sample((n_samples, B), conditions=enc)  # [n_samples, B, K*p]
