"""Horizon sweep: worst-corner sPCE vs horizon T, per (policy, seed).

Run: uv run python scripts/horizon.py train [--policies ...] [--t-values ...] [--seeds 0 1]
     uv run python scripts/horizon.py eval [--seeds 0 1] [--b-eval 1000] [--l-contrastive 10000]
     uv run python scripts/horizon.py plot [--results output/horizon/results.json]

Training is resumable: existing checkpoints are skipped; delete a .pt to force a retrain.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from stochastic_bed.eig import spce_bound
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
EVAL_NAMES = (*POLICY_ORDER, "random")
PRIOR_MEANS = [(1.5, 1.5), (1.5, -1.5), (-1.5, 1.5), (-1.5, -1.5)]
DESIGN_BOUND = 3.0
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "horizon"

STYLES = {
    "deterministic": "tab:blue",
    "stochastic": "tab:purple",
    "static_deterministic": "tab:cyan",
    "static_stochastic": "tab:pink",
    "random": "tab:red",
}
GAP_PAIRS = [
    ("adaptive", "stochastic", "deterministic", "black"),
    ("static", "static_stochastic", "static_deterministic", "tab:gray"),
]


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


def train_cmd(args):
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


def eval_cmd(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = []
    with torch.no_grad():
        for T in args.t_values:
            sims = make_simulators(T)
            for j, name in enumerate(EVAL_NAMES):
                for s in args.seeds if name != "random" else [0]:
                    policy = build_policy(name, sims[0]).to(device)
                    if name != "random":
                        ckpt = args.dir / f"T{T}_{name}_s{s}.pt"
                        if not ckpt.exists():
                            print(f"T={T} {name} s{s}: no checkpoint, skipping", flush=True)
                            continue
                        policy.load_state_dict(torch.load(ckpt, map_location=device))
                    scores = []
                    for i, member in enumerate(sims):
                        # Seed per (T, policy, corner), shared across training seeds (paired eval)
                        torch.manual_seed(100_000 + 1000 * T + 10 * j + i)
                        theta = member.prior().sample((args.b_eval,)).to(device)
                        theta_c = (
                            member.prior().sample((args.b_eval, args.l_contrastive)).to(device)
                        )
                        traj = member.rollout(theta, policy)
                        spce = spce_bound(member, theta, traj.designs, traj.outcomes, theta_c)
                        scores.append(spce.mean().item())
                    records.append(
                        {
                            "T": T,
                            "policy": name,
                            "seed": s,
                            "per_corner": scores,
                            "worst": min(scores),
                        }
                    )
                    print(f"T={T} {name} s{s}: worst {min(scores):.3f}", flush=True)

    out = args.dir / "results.json"
    out.write_text(json.dumps({"b_eval": args.b_eval, "L": args.l_contrastive, "records": records}))
    print(f"wrote {out} ({len(records)} records)")


def plot_cmd(args):
    records = json.loads(args.results.read_text())["records"]
    worst = {}  # (T, policy) -> {seed: worst score}
    for r in records:
        worst.setdefault((r["T"], r["policy"]), {})[r["seed"]] = r["worst"]
    t_values = sorted({r["T"] for r in records})

    fig, (ax_worst, ax_gap) = plt.subplots(1, 2, figsize=(11, 4))
    for name, color in STYLES.items():
        Ts = [T for T in t_values if (T, name) in worst]
        if Ts:
            vals = [list(worst[T, name].values()) for T in Ts]
            ax_worst.plot(Ts, [sum(v) / len(v) for v in vals], marker="o", label=name, color=color)
            ax_worst.fill_between(
                Ts, [min(v) for v in vals], [max(v) for v in vals], color=color, alpha=0.2
            )
    ax_worst.set_xlabel("training/eval horizon T")
    ax_worst.set_ylabel("worst-corner final sPCE")
    ax_worst.set_title("robust EIG vs horizon (policies retrained per T)")
    ax_worst.legend()

    for label, sto, det, color in GAP_PAIRS:
        Ts = [T for T in t_values if (T, sto) in worst and (T, det) in worst]
        if Ts:
            gaps = [
                [worst[T, sto][s] - worst[T, det][s] for s in worst[T, sto] if s in worst[T, det]]
                for T in Ts
            ]
            ax_gap.plot(Ts, [sum(g) / len(g) for g in gaps], marker="o", color=color, label=label)
            ax_gap.fill_between(
                Ts, [min(g) for g in gaps], [max(g) for g in gaps], color=color, alpha=0.2
            )
    ax_gap.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax_gap.set_xlabel("training/eval horizon T")
    ax_gap.set_ylabel("stochastic − deterministic (worst-corner sPCE)")
    ax_gap.set_title("value of mixing vs horizon")
    ax_gap.legend()
    fig.tight_layout()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "horizon.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("train")
    p.add_argument("--policies", nargs="+", default=list(POLICY_ORDER), choices=POLICY_ORDER)
    p.add_argument("--t-values", nargs="+", type=int, default=list(range(1, 9)))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--num-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dir", type=Path, default=OUTPUT_DIR)
    p.set_defaults(func=train_cmd)

    p = sub.add_parser("eval")
    p.add_argument("--t-values", nargs="+", type=int, default=list(range(1, 9)))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--b-eval", type=int, default=1000)
    p.add_argument("--l-contrastive", type=int, default=10000)
    p.add_argument("--dir", type=Path, default=OUTPUT_DIR)
    p.set_defaults(func=eval_cmd)

    p = sub.add_parser("plot")
    p.add_argument("--results", type=Path, default=OUTPUT_DIR / "results.json")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR.parent / "figures")
    p.set_defaults(func=plot_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
