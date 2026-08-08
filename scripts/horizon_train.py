"""Horizon sweep, training stage: retrain per-(T, policy) GDRO checkpoints into output/horizon/.

Run: uv run python scripts/horizon_train.py [--policies ...] [--t-values 1 2 ...] [--seeds 0 1 2]
Resumable: existing checkpoints are skipped; delete a .pt to force a retrain.
"""

import argparse
import json
from pathlib import Path

import torch

from stochastic_bed.policies import (
    DeterministicPolicy,
    RandomPolicy,
    StaticDeterministicPolicy,
    StaticStochasticPolicy,
    StochasticPolicy,
)
from stochastic_bed.posterior import PosteriorNet
from stochastic_bed.simulator import LocationFinding
from stochastic_bed.train import train

# Seed index per policy: extend only at the end, so cached runs keep their seeds
POLICY_ORDER = ("stochastic", "deterministic", "static_stochastic", "static_deterministic")
PRIOR_MEANS = [(1.5, 1.5), (1.5, -1.5), (-1.5, 1.5), (-1.5, -1.5)]
DESIGN_BOUND = 3.0
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "horizon"


def make_simulators(T):
    return [
        LocationFinding(prior_mean=torch.tensor(m), prior_std=0.5, a=0.25, T=T) for m in PRIOR_MEANS
    ]


def build_policy(name, sim):
    match name:
        case "stochastic":
            return StochasticPolicy(D=sim.D, p=sim.p, design_bound=DESIGN_BOUND)
        case "deterministic":
            return DeterministicPolicy(D=sim.D, p=sim.p, design_bound=DESIGN_BOUND)
        case "static_stochastic":
            return StaticStochasticPolicy(D=sim.D, p=sim.p, design_bound=DESIGN_BOUND, T=sim.T)
        case "static_deterministic":
            return StaticDeterministicPolicy(D=sim.D, p=sim.p, design_bound=DESIGN_BOUND, T=sim.T)
        case "random":
            return RandomPolicy(D=sim.D, p=sim.p, design_bound=DESIGN_BOUND)
    raise ValueError(f"Unknown policy: {name!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", default=list(POLICY_ORDER), choices=POLICY_ORDER)
    parser.add_argument("--t-values", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    args.dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for T in args.t_values:
        for name in args.policies:
            for s in args.seeds:
                ckpt = args.dir / f"T{T}_{name}_s{s}.pt"
                if ckpt.exists():
                    print(f"T={T} {name} s{s}: cached, skipping", flush=True)
                    continue
                try:
                    torch.manual_seed(1_000_000 * s + 1000 * T + POLICY_ORDER.index(name))
                    sims = make_simulators(T)
                    policy = build_policy(name, sims[0])
                    posterior = PosteriorNet(D=sims[0].D, p=sims[0].p, K=sims[0].K)
                    metrics = train(
                        sims,
                        policy,
                        posterior,
                        num_steps=args.num_steps,
                        batch_size=args.batch_size,
                        verbose=False,
                    )
                    torch.save(policy.state_dict(), ckpt)
                    (args.dir / f"T{T}_{name}_s{s}_metrics.json").write_text(json.dumps(metrics))
                    weights = " ".join(f"{w:.2f}" for w in metrics["weights"][-1])
                    print(
                        f"T={T} {name} s{s}: loss {metrics['loss'][-1]:.4f}  weights [{weights}]",
                        flush=True,
                    )
                except Exception as e:
                    failures.append((T, name, s))
                    print(f"T={T} {name} s{s}: FAILED — {e!r}", flush=True)

    print(f"done; failures: {failures}" if failures else "done")


if __name__ == "__main__":
    main()
