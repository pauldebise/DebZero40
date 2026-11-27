"""Trace les résultats du model soup nano_ib48 depuis le JSON d'évaluation."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def virgule(x, _):
    return f"{x:.2f}".replace(".", ",")


def pourcent(x, _):
    return f"{x:.1f} %".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/mesures/soup_nano_ib48.json")
    ap.add_argument("--out", default="docs/mesures/soup_nano_ib48.png")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as fh:
        data = json.load(fh)
    alphas = data["alphas"]
    res = data["resultats"]
    cles = [f"alpha_{a:.2f}".replace(".", "_") for a in alphas]
    policy = [res[k]["policy_loss"] for k in cles]
    wdl = [res[k]["wdl_loss"] for k in cles]
    top1 = [100 * res[k]["top1"] for k in cles]
    wdl_acc = [100 * res[k]["wdl_acc"] for k in cles]
    brier = [res[k]["brier"] for k in cles]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), dpi=160)
    fig.suptitle("Model soup nano_ib48 : interpolation best (ép. 15) → last (ép. 16)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")

    # 1. Pertes policy et WDL, deux axes.
    ax = axes[0][0]
    ax.plot(alphas, policy, "o-", color="#1f6feb", lw=1.8, ms=5, label="perte politique")
    ax.set_xlabel("alpha (poids de last)")
    ax.set_ylabel("perte politique", color="#1f6feb")
    ax.tick_params(axis="y", labelcolor="#1f6feb")
    ax.yaxis.set_major_formatter(FuncFormatter(virgule))
    ax2 = ax.twinx()
    ax2.plot(alphas, wdl, "s-", color="#d1541f", lw=1.8, ms=5, label="perte WDL")
    ax2.set_ylabel("perte WDL", color="#d1541f")
    ax2.tick_params(axis="y", labelcolor="#d1541f")
    ax2.yaxis.set_major_formatter(FuncFormatter(virgule))
    ax.set_title("Pertes de validation", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.2, ls=":")
    lignes = ax.get_lines() + ax2.get_lines()
    ax.legend(lignes, [l.get_label() for l in lignes], frameon=False, fontsize=8,
              loc="upper left")

    # 2. Top-1.
    ax = axes[0][1]
    ax.plot(alphas, top1, "o-", color="#1a7f37", lw=1.8, ms=5)
    ax.set_xlabel("alpha (poids de last)")
    ax.set_ylabel("top-1 politique")
    ax.yaxis.set_major_formatter(FuncFormatter(pourcent))
    ax.set_ylim(42.4, 43.7)
    ax.set_title("Top-1 politique", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.25, ls=":")
    for a, v in zip(alphas, top1):
        ax.annotate(f"{v:.2f} %".replace(".", ","), (a, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)

    # 3. Exactitude WDL.
    ax = axes[1][0]
    ax.plot(alphas, wdl_acc, "s-", color="#bf8700", lw=1.8, ms=5)
    ax.set_xlabel("alpha (poids de last)")
    ax.set_ylabel("exactitude WDL")
    ax.yaxis.set_major_formatter(FuncFormatter(pourcent))
    ax.set_ylim(82.8, 86.4)
    ax.set_title("Exactitude WDL", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.25, ls=":")
    for a, v in zip(alphas, wdl_acc):
        ax.annotate(f"{v:.2f} %".replace(".", ","), (a, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)

    # 4. Brier.
    ax = axes[1][1]
    ax.plot(alphas, brier, "^-", color="#8250df", lw=1.8, ms=5)
    ax.set_xlabel("alpha (poids de last)")
    ax.set_ylabel("score de Brier")
    ax.yaxis.set_major_formatter(FuncFormatter(virgule))
    ax.set_title("Score de Brier (plus bas = mieux)", fontsize=11,
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.25, ls=":")
    for a, v in zip(alphas, brier):
        ax.annotate(f"{v:.3f}".replace(".", ","), (a, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)

    # Repère du compromis retenu.
    for ax in axes.flat:
        ax.axvline(0.5, color="#8c959f", ls="--", lw=1, alpha=0.7)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks(alphas)
        ax.xaxis.set_major_formatter(FuncFormatter(virgule))
    fig.text(0.5, 0.005, "trait vertical : alpha = 0,5, compromis retenu",
             ha="center", fontsize=8, color="#57606a")

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
