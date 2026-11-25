"""Trace les courbes d'entraînement à partir de ``history.jsonl``."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def charger(path):
    lignes = []
    with open(path, encoding="utf-8") as fh:
        for ligne in fh:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))
    return lignes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--out", default="docs/courbes_nano.png")
    ap.add_argument("--titre", default="Entraînement nano (distillation LC0)")
    args = ap.parse_args()

    h = charger(args.history)
    ep = [x["epoch"] for x in h]
    tr = [x["train"] for x in h]
    va = [x["val"] for x in h]
    lr = [x["lr"] for x in h]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(args.titre, fontsize=13)

    ax = axes[0][0]
    ax.plot(ep, [t["loss"] for t in tr], "o-", label="entraînement")
    if va[0] is not None:
        ax.plot(ep, [v["loss"] for v in va], "s-", label="validation")
    ax.set_title("Perte totale")
    ax.set_xlabel("époque")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(ep, [t["policy_loss"] for t in tr], "o-", label="policy (train)")
    ax.plot(ep, [t["wdl_loss"] for t in tr], "o-", label="wdl (train)")
    if va[0] is not None:
        ax.plot(ep, [v["policy_loss"] for v in va], "s--", label="policy (val)")
        ax.plot(ep, [v["wdl_loss"] for v in va], "s--", label="wdl (val)")
    ax.set_title("Décomposition de la perte")
    ax.set_xlabel("époque")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(ep, [t["top1"] * 100 for t in tr], "o-", label="top-1 policy (train)")
    if va[0] is not None:
        ax.plot(ep, [v["top1"] * 100 for v in va], "s-", label="top-1 policy (val)")
        ax2 = ax.twinx()
        ax2.plot(ep, [v["wdl_acc"] * 100 for v in va], "^:", color="gray", label="WDL (val)")
        ax2.set_ylabel("exactitude WDL (%)")
        ax2.legend(fontsize=8, loc="lower right")
    ax.set_title("Qualité : top-1 policy et WDL")
    ax.set_xlabel("époque")
    ax.set_ylabel("top-1 (%)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(ep, lr, "o-", color="purple")
    ax.set_title("Learning rate (cosine)")
    ax.set_xlabel("époque")
    ax.ticklabel_format(axis="y", style="sci")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"courbes écrites : {args.out} ({len(h)} époques)")


if __name__ == "__main__":
    main()
