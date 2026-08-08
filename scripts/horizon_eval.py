"""Horizon sweep, eval stage: worst-corner sPCE per (T, policy) checkpoint -> results.json.

Run: uv run python scripts/horizon_eval.py [--b-eval 1000] [--l-contrastive 10000] [--seeds 0 1 2]
Scores every checkpoint found in --dir; missing ones are skipped.
"""

import argparse
import json
from pathlib import Path

import torch

from horizon_train import OUTPUT_DIR, POLICY_ORDER, build_policy, make_simulators
from stochastic_bed.eig import spce_bound

EVAL_NAMES = (*POLICY_ORDER, "random")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-values", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--b-eval", type=int, default=1000)
    parser.add_argument("--l-contrastive", type=int, default=10000)
    parser.add_argument("--dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
