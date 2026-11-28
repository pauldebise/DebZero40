"""Courbes détaillées d'un run v3 (et comparaison avec un run v2).

Le fichier d'entrée est le ``history.jsonl`` écrit par ``src/train/train.py`` :
une ligne JSON par époque, avec les pertes, le top-1, l'exactitude WDL et le LR.

Exemple :
    python bench/plot_history_v3.py \
        --history docs/mesures/nano_ib48_history.jsonl \
        --v2 nets/v2/nano_full_history.jsonl \
        --screen docs/mesures \
        --out docs/mesures/courbes_nano_ib48.png
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


TRAIN = "#1f6feb"
VAL = "#d1541f"
WDL = "#8250df"
GRID = "#8c959f"


def charger(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def virgule(x, _pos=None):
    return f"{x:g}".replace(".", ",")


def milliers(x, _pos=None):
    return f"{x:,.0f}".replace(",", "\u202f")


def style(ax, titre, ylabel=None, xlabel="époque", yformatter=True):
    ax.set_title(titre, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.22, ls=":", color=GRID)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(virgule))
    if yformatter:
        ax.yaxis.set_major_formatter(FuncFormatter(virgule))


def courbes_run(h, out, titre, externe=None):
    ep = [r["epoch"] for r in h]
    tr = [r["train"] for r in h]
    va = [r["val"] for r in h]
    lr = [r["lr"] for r in h]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), dpi=150)
    fig.suptitle(titre, fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.925, "111 790 paramètres  ·  6,35 M opérations  ·  0,46 ms par position sur le VPS  ·  "
             "distillation LC0, aucun auto-jeu", fontsize=9, color=GRID, ha="left")

    # 1. Perte totale
    ax = axes[0][0]
    ax.plot(ep, [t["loss"] for t in tr], "o-", color=TRAIN, lw=1.8, ms=4, label="entraînement")
    ax.plot(ep, [v["loss"] for v in va], "s-", color=VAL, lw=1.8, ms=4, label="validation")
    style(ax, "Perte totale", "perte")
    ax.legend(frameon=False, fontsize=8)
    ax.annotate("la validation est dominée\npar l'oscillation du WDL",
                xy=(1, va[0]["loss"]), xytext=(4.4, 5.65), fontsize=8,
                color=VAL, arrowprops=dict(arrowstyle="->", color=VAL, lw=1))

    # 2. Perte de politique
    ax = axes[0][1]
    ax.plot(ep, [t["policy_loss"] for t in tr], "o-", color=TRAIN, lw=1.8, ms=4, label="entraînement")
    ax.plot(ep, [v["policy_loss"] for v in va], "s-", color=VAL, lw=1.8, ms=4, label="validation")
    style(ax, "Perte de politique (KL cible → réseau)", "perte")
    ax.legend(frameon=False, fontsize=8)
    ax.annotate(f"{va[-1]['policy_loss']:.4f}".replace(".", ","),
                xy=(ep[-1], va[-1]["policy_loss"]), xytext=(ep[-1] - 3.2, va[-1]["policy_loss"] + 0.018),
                fontsize=8, color=VAL, va="bottom")

    # 3. Perte WDL
    ax = axes[0][2]
    ax.plot(ep, [t["wdl_loss"] for t in tr], "o-", color=TRAIN, lw=1.8, ms=4, label="entraînement")
    ax.plot(ep, [v["wdl_loss"] for v in va], "s-", color=WDL, lw=1.8, ms=4, label="validation")
    ax.axhline(1.0986, color=GRID, ls="--", lw=1, label="hasard (ln 3)")
    style(ax, "Perte WDL (victoire / nulle / défaite)", "perte")
    ax.set_ylim(0.6, 2.65)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    pire = max(range(len(va)), key=lambda i: va[i]["wdl_loss"])
    ax.annotate(f"jusqu'à {va[pire]['wdl_loss']:.2f} à l'époque {ep[pire]} :\nau-delà du hasard",
                xy=(ep[pire], va[pire]["wdl_loss"]),
                xytext=(4.4, 2.32), fontsize=8, color=WDL,
                arrowprops=dict(arrowstyle="->", color=WDL, lw=1))

    # 4. Top-1 politique
    ax = axes[1][0]
    ax.plot(ep, [100 * t["top1"] for t in tr], "o-", color=TRAIN, lw=1.8, ms=4, label="entraînement")
    ax.plot(ep, [100 * v["top1"] for v in va], "s-", color=VAL, lw=1.8, ms=4, label="validation")
    ax.axhline(35.24, color=GRID, ls=":", lw=1.2)
    ax.text(1.2, 35.9, "meilleur nano v2 : 35,2 %", fontsize=8, color=GRID)
    style(ax, "Top-1 de la politique", "top-1 (%)")
    if externe is not None:
        ax.scatter([ep[-1] + 0.18], [100 * externe], marker="*", s=150, color="#1a7f37", zorder=5,
                   label="évaluation externe")
        ax.annotate(f"externe : {100 * externe:.2f} %".replace(".", ","),
                    xy=(ep[-1] + 0.18, 100 * externe), xytext=(11.4, 100 * externe + 1.4),
                    fontsize=8, color="#1a7f37",
                    arrowprops=dict(arrowstyle="->", color="#1a7f37", lw=1))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.annotate(f"{100 * va[-1]['top1']:.2f} %".replace(".", ","),
                xy=(ep[-1], 100 * va[-1]["top1"]), xytext=(ep[-1] - 3.1, 100 * va[-1]["top1"] - 1.4),
                fontsize=8, color=VAL)

    # 5. Exactitude WDL
    ax = axes[1][1]
    ax.plot(ep, [100 * t["wdl_acc"] for t in tr], "o-", color=TRAIN, lw=1.8, ms=4, label="entraînement")
    ax.plot(ep, [100 * v["wdl_acc"] for v in va], "s-", color=WDL, lw=1.8, ms=4, label="validation")
    style(ax, "Exactitude WDL", "exactitude (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # 6. Learning rate
    ax = axes[1][2]
    ax.plot(ep, lr, "o-", color="#57606a", lw=1.8, ms=4)
    ax.set_yscale("log")
    style(ax, "Learning rate (cosine)", "LR")
    ax.annotate("0 en fin de run", xy=(ep[-1], max(lr[-1], 1e-9)), xytext=(10.5, 2e-4),
                fontsize=8, color="#57606a",
                arrowprops=dict(arrowstyle="->", color="#57606a", lw=1))

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"écrit : {out}")


def comparaison(h3, h2, out):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=150)
    fig.suptitle("Nano v3 (nano_ib48) contre nano v2 — mêmes données, même recette",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")

    ax = axes[0]
    ax.plot([r["epoch"] for r in h2], [100 * r["val"]["top1"] for r in h2],
            "s--", color=GRID, lw=1.6, ms=3, label="nano v2 (2×32, 312 k param.)")
    ax.plot([r["epoch"] for r in h3], [100 * r["val"]["top1"] for r in h3],
            "o-", color=VAL, lw=2, ms=4, label="nano v3 (ib48, 112 k param.)")
    style(ax, "Top-1 de validation", "top-1 (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.annotate("+8,1 points", xy=(16, 43.34), xytext=(10.8, 39.3), fontsize=9,
                color=VAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VAL, lw=1.2))

    ax = axes[1]
    ax.plot([r["epoch"] for r in h2], [r["val"]["policy_loss"] for r in h2],
            "s--", color=GRID, lw=1.6, ms=3, label="nano v2")
    ax.plot([r["epoch"] for r in h3], [r["val"]["policy_loss"] for r in h3],
            "o-", color=VAL, lw=2, ms=4, label="nano v3")
    style(ax, "Perte de politique (validation)", "perte")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # Criblage : moyenne des graines disponibles, points individuels.
    ax = axes[2]
    valeurs = defaultdict(list)
    if os.path.isdir(os.path.join(os.path.dirname(out), "..")):
        base = os.path.dirname(out)
        for graine in ("screen1", "screen2"):
            dossier = os.path.join(base, graine)
            if not os.path.isdir(dossier):
                continue
            for nom in sorted(os.listdir(dossier)):
                hist = os.path.join(dossier, nom, "history.jsonl")
                if not os.path.isfile(hist):
                    continue
                rows = charger(hist)
                if rows:
                    valeurs[nom.replace("sc_", "")].append(100 * rows[-1]["val"]["top1"])
    if valeurs:
        noms = sorted(valeurs, key=lambda n: sum(valeurs[n]) / len(valeurs[n]), reverse=True)
        moyens = [sum(valeurs[n]) / len(valeurs[n]) for n in noms]
        couleurs = ["#1a7f37" if n == "nano_ib48" else "#8c959f" for n in noms]
        ax.barh(range(len(noms)), moyens, color=couleurs, height=0.62)
        for i, n in enumerate(noms):
            ax.scatter(valeurs[n], [i] * len(valeurs[n]), color="#24292f", s=22, zorder=3)
            ax.text(moyens[i] + 0.25, i, f"{moyens[i]:.1f}".replace(".", ","),
                    va="center", fontsize=8)
        ax.set_yticks(range(len(noms)))
        ax.set_yticklabels(noms, fontsize=8)
        ax.invert_yaxis()
        style(ax, "Criblage à 2 époques (deux graines)", "top-1 validation (%)", "", yformatter=False)
        ax.set_xlim(27, 35)
        ax.set_ylim(len(noms) - 0.45, -0.55)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#1a7f37", label="finaliste"),
                           Patch(color="#8c959f", label="éliminé"),
                           plt.Line2D([], [], marker="o", ls="", color="#24292f", label="graine")],
                  frameon=False, fontsize=7.5, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"écrit : {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--v2", default="")
    ap.add_argument("--out", default="docs/mesures/courbes_nano_ib48.png")
    ap.add_argument("--out-compare", default="")
    ap.add_argument("--titre", default="nano_ib48 — 16 époques sur 20 M de positions (LC0)")
    ap.add_argument("--externe", type=float, default=None,
                    help="top-1 externe (fraction 0-1) à marquer d'une étoile ; omis si absent")
    args = ap.parse_args()

    h = charger(args.history)
    courbes_run(h, args.out, args.titre, args.externe)
    if args.v2 and args.out_compare:
        comparaison(h, charger(args.v2), args.out_compare)


if __name__ == "__main__":
    main()
