"""Train policy and posterior jointly with adversary-weighted Barber-Agakov loss"""

from collections import defaultdict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .posterior import PosteriorNet
from .simulator import LocationFinding


def train(
    simulators: list[LocationFinding],
    policy: nn.Module,
    posterior: PosteriorNet,
    *,
    num_steps: int = 3000,
    batch_size: int = 256,
    max_lr: float = 1e-3,
    clip_norm: float | None = 2.0,
    adversary_step_size: float = 0.01,
    device: torch.device | None = None,
    verbose: bool = True,
    print_every: int = 50,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy.to(device)
    posterior.to(device)

    trainable = nn.ModuleList([policy, posterior])
    trainable.train()
    optimizer = torch.optim.AdamW(trainable.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=num_steps,
        pct_start=0.4,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=25,
    )

    n_groups = len(simulators)
    group_batch_size = batch_size // n_groups
    weights = torch.full((n_groups,), 1.0 / n_groups, device=device)

    if clip_norm is None:
        clip_norm = float("inf")

    metrics = defaultdict(list)

    pbar = tqdm(range(num_steps), disable=not verbose)

    with torch.enable_grad():  # Bayesflow's torch backend disables autograd globally
        for step in pbar:
            # Group DRO (Sagawa et al., 2020): adversary upweights high-loss groups, model minimizes weighted loss
            group_losses, stds = [], []
            for sim in simulators:
                theta = sim.prior().sample((group_batch_size,)).to(device)
                traj = sim.rollout(theta, policy)

                group_losses.append(posterior.loss(theta, traj.designs, traj.outcomes).mean())
                if traj.stds is not None:
                    stds.append(traj.stds.mean(dim=(0, 2, 3)))  # [T]
            group_losses = torch.stack(group_losses)

            with torch.no_grad():
                weights = torch.softmax(weights.log() + adversary_step_size * group_losses, dim=0)
            loss = weights @ group_losses

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), clip_norm)

            lr = scheduler.get_last_lr()[0]
            # A single non-finite grad would NaN all params via the clip
            if torch.isfinite(grad_norm):
                optimizer.step()
            elif verbose:
                tqdm.write(f"Step {step}: non-finite grad norm, skipping optimizer step")
            scheduler.step()

            step_metrics = {
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "learning_rate": lr,
                "group_losses": group_losses.detach().tolist(),
                "weights": weights.tolist(),
                "stds": torch.stack(stds).mean(dim=0).tolist() if stds else None,
            }
            for key, value in step_metrics.items():
                if value is not None:
                    metrics[key].append(value)

            formatted = {
                key: f"{value:.4g}"
                for key, value in step_metrics.items()
                if value is not None and not isinstance(value, list)
            }
            if n_groups > 1:
                formatted |= {
                    "worst_group": str(int(group_losses.argmax())),
                    "max_weight": f"{weights.max().item():.3f}",
                }
            pbar.set_postfix(formatted)
            if verbose and step % print_every == 0:
                tqdm.write(f"Step {step}: " + "   ".join(f"{k} {v}" for k, v in formatted.items()))

    return dict(metrics)
