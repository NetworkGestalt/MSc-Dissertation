"""Horizon sweep, plot stage: robust-EIG and value-of-mixing figures from results.json.

Run: uv run python scripts/horizon_plot.py [--results output/horizon/results.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from horizon_train import OUTPUT_DIR

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUTPUT_DIR / "results.json")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR.parent / "figures")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
