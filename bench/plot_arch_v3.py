"""Schéma de l'architecture nano_ib48 (gamme v3), pour la documentation.

    python bench/plot_arch_v3.py --out docs/mesures/arch_nano_ib48.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLEU = "#1f6feb"
VIOLET = "#8250df"
VERT = "#1a7f37"
ORANGE = "#d1541f"
GRIS = "#57606a"
FOND_BLEU = "#ddf4ff"
FOND_VIOLET = "#f3e8ff"
FOND_VERT = "#dafbe1"
FOND_ORANGE = "#fff1e5"
FOND_GRIS = "#f6f8fa"


def boite(ax, x, y, w, h, titre, corps, bord, fond, taille_titre=10, taille_corps=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.4, edgecolor=bord, facecolor=fond))
    ax.text(x + w / 2, y + h - 0.045, titre, ha="center", va="top",
            fontsize=taille_titre, fontweight="bold", color=bord)
    ax.text(x + w / 2, y + h - 0.135, corps, ha="center", va="top",
            fontsize=taille_corps, color="#24292f", linespacing=1.35)


def fleche(ax, x1, y1, x2, y2, couleur=GRIS):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.5, color=couleur, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/mesures/arch_nano_ib48.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(16, 8.2), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    ax.text(0.25, 7.95, "nano_ib48 — l'architecture du dernier run",
            fontsize=17, fontweight="bold", color="#24292f", va="top")
    ax.text(0.25, 7.55, "111 790 paramètres  ·  6,35 M opérations  ·  0,46 ms par position  ·  "
            "entrée 17 plans, sortie politique 1 858 coups + WDL",
            fontsize=10, color=GRIS, va="top")

    # Entrée
    boite(ax, 0.25, 4.55, 1.85, 2.15, "Entrée", "17 × 8 × 8\n\n12 plans de pièces\n"
          "4 droits de roque\n1 compteur règle 50\n\nvue du joueur au trait", GRIS, FOND_GRIS,
          taille_titre=10.5)
    fleche(ax, 2.10, 5.62, 2.55, 5.62)

    # Tige
    boite(ax, 2.55, 4.85, 1.75, 1.55, "Tige", "Conv 3×3\n17 → 48\nBN + SiLU\n\n7 440 param.",
          BLEU, FOND_BLEU)
    fleche(ax, 4.30, 5.62, 4.75, 5.62)

    # Tronc
    ax.add_patch(FancyBboxPatch((4.75, 3.35), 5.35, 3.35,
                                boxstyle="round,pad=0.015,rounding_size=0.03",
                                linewidth=1.8, edgecolor=VIOLET, facecolor=FOND_VIOLET))
    ax.text(7.42, 6.52, "6 × bloc à goulot inversé (48 canaux)", ha="center", va="top",
            fontsize=11, fontweight="bold", color=VIOLET)
    ax.text(7.42, 6.28, "même bloc répété six fois, résidu à chaque fois", ha="center",
            va="top", fontsize=8.2, color=GRIS, style="italic")

    # Sous-boîtes du bloc
    y0, h0 = 4.45, 1.55
    boite(ax, 4.95, y0, 1.05, h0, "DW 5×5", "48 → 48\nprofonde\n\n1 200 p.", VIOLET, "white", 9, 8)
    fleche(ax, 6.00, y0 + h0 / 2, 6.20, y0 + h0 / 2)
    boite(ax, 6.20, y0, 1.05, h0, "PW 1×1", "48 → 144\n×3 canaux\n\n6 912 p.", VIOLET, "white", 9, 8)
    fleche(ax, 7.25, y0 + h0 / 2, 7.45, y0 + h0 / 2)
    boite(ax, 7.45, y0, 1.05, h0, "PW 1×1", "144 → 48\nprojection\n\n6 912 p.", VIOLET, "white", 9, 8)
    fleche(ax, 8.50, y0 + h0 / 2, 8.70, y0 + h0 / 2)
    boite(ax, 8.70, y0, 1.15, h0, "SE 8×", "48 → 6 → 48\npoids par canal\n\n630 p.", VIOLET, "white", 9, 8)

    ax.text(7.42, 4.08, "BN + SiLU après chaque convolution  ·  sortie = entrée + bloc(entrée)  ·  "
            "16 038 param. par bloc, 96 228 au total",
            ha="center", va="center", fontsize=8.3, color="#24292f")
    ax.text(7.42, 3.62, "pas de contexte global dans le nano  ·  pas de changement de résolution",
            ha="center", va="center", fontsize=8.3, color=GRIS, style="italic")

    # Flèche vers les têtes
    fleche(ax, 10.10, 5.62, 10.55, 5.62)
    ax.plot([10.55, 10.55], [2.55, 6.55], color=GRIS, lw=1.5)
    fleche(ax, 10.55, 6.35, 10.95, 6.35, VERT)
    fleche(ax, 10.55, 2.75, 10.95, 2.75, ORANGE)

    # Tête politique
    boite(ax, 10.95, 5.35, 4.80, 2.05, "Tête politique bilinéaire",
          "1×1 48 → 32  + BN + SiLU\n"
          "q et k : 32 → 32, appliqués aux 64 cases\n"
          "score(départ, arrivée) = q · k  →  matrice 64 × 64\n"
          "on ne garde que les 1 858 coups LC0  + biais de promotion\n"
          "masque des coups légaux → softmax\n\n3 652 param.", VERT, FOND_VERT, 10, 8.2)
    ax.text(13.35, 5.08, "Sortie : 1 858 probabilités", ha="center", va="center",
            fontsize=8.6, color=VERT, fontweight="bold")

    # Tête de valeur
    boite(ax, 10.95, 1.75, 4.80, 1.85, "Tête de valeur",
          "1×1 48 → 1  + BN + SiLU\n"
          "aplatir 64 cases → FC 64 → FC 4\n"
          "3 logits WDL → softmax  +  1 sortie MLH → softplus\n\n4 470 param.", ORANGE, FOND_ORANGE, 10, 8.2)
    ax.text(13.35, 1.50, "Sortie : WDL (3)  +  MLH (1, inerte)", ha="center", va="center",
            fontsize=8.6, color=ORANGE, fontweight="bold")

    # Bandeau des trois choix
    ax.add_patch(FancyBboxPatch((0.25, 0.25), 15.5, 1.05,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor="#d0d7de", facecolor="#f6f8fa"))
    ax.text(0.45, 1.06, "Les trois paris de la gamme v3", fontsize=9.5, fontweight="bold",
            color="#24292f", va="top")
    ax.text(0.45, 0.78,
            "1. Goulot inversé : à opérations égales, plus de canaux — mais la convolution profonde coûte ~2× plus cher par opération sur ce CPU.\n"
            "2. Tête bilinéaire : le logit d'un coup est un produit scalaire départ · arrivée ; 3 652 paramètres, pas 450 000 comme une tête dense.\n"
            "3. WDL + MLH : forcer le tronc à encoder la phase de jeu ; le MLH est inerte ici car les shards n'ont pas de cible « coups restants ».",
            fontsize=8.2, color="#24292f", va="top", linespacing=1.45)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
