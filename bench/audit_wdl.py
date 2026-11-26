"""Audit des étiquettes WDL dans les shards d'entraînement.

Vérifie que la cible ``wdl`` est bien exprimée **du point de vue du joueur au trait**
et cohérente avec le matériel. Les plans 0-5 sont nos pièces, 6-11 celles de
l'adversaire : le déséquilibre matériel se lit donc directement dans le tenseur.

Une étiquette saine vérifie :
- à matériel équilibré, P(gain) ≈ 0,5 (par symétrie) ;
- plus on a de matériel, plus P(gain) monte. Une corrélation nulle ou négative
  signale des étiquettes retournées.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

#: Valeur des pièces, indexée par type (les plans suivent l'ordre des pièces de `chess`).
VALEUR = np.array([0, 1, 3, 3, 5, 9, 0], dtype=np.float32)


def stats_shard(path: str, max_positions: int = 50000) -> dict:
    with np.load(path) as d:
        planes = d["planes"][:max_positions]
        wdl = d["wdl"][:max_positions].astype(np.float32)

    if planes.dtype == np.uint8:
        planes = planes.astype(np.float32)
    if planes.ndim != 4:
        raise ValueError(f"{path}: plans de forme inattendue {planes.shape}")

    n = min(len(planes), max_positions)
    plan = planes[:n]
    wdl = wdl[:n]

    # Matériel du point de vue du joueur au trait.
    ours = plan[:, 0:6].sum(axis=(2, 3)) @ VALEUR[1:7]
    theirs = plan[:, 6:12].sum(axis=(2, 3)) @ VALEUR[1:7]
    diff = ours - theirs

    p_win = wdl[:, 0]
    p_loss = wdl[:, 2]
    equilibre = np.abs(diff) < 0.5
    avance = diff >= 3.0
    retard = diff <= -3.0

    return {
        "shard": os.path.basename(path),
        "n": int(n),
        "p_win_moy": float(p_win.mean()),
        "p_win_equilibre": float(p_win[equilibre].mean()) if equilibre.any() else float("nan"),
        "p_win_avance": float(p_win[avance].mean()) if avance.any() else float("nan"),
        "p_win_retard": float(p_win[retard].mean()) if retard.any() else float("nan"),
        "corr_materiel": float(np.corrcoef(diff, p_win)[0, 1]) if n > 10 else float("nan"),
        "part_equilibre": float(equilibre.mean()),
        "regression": float(np.abs(p_win + wdl[:, 1] + p_loss - 1).max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="répertoire des .npz")
    ap.add_argument("--max-positions", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=0, help="ne traiter que les N premiers shards")
    args = ap.parse_args()

    fichiers = sorted(glob.glob(os.path.join(args.shards, "*.npz")))
    if args.limit:
        fichiers = fichiers[: args.limit]
    print(f"{len(fichiers)} shards à auditer\n")
    print(f"{'shard':22s} {'n':>7s} {'P(gain)':>8s} {'équil.':>7s} {'avance':>7s} {'retard':>7s} {'corr':>6s}")

    lignes = []
    for f in fichiers:
        s = stats_shard(f, args.max_positions)
        lignes.append(s)
        print(
            f"{s['shard']:22s} {s['n']:7d} {s['p_win_moy']:8.3f} {s['p_win_equilibre']:7.3f} "
            f"{s['p_win_avance']:7.3f} {s['p_win_retard']:7.3f} {s['corr_materiel']:+6.2f}",
            flush=True,
        )

    corr = np.array([s["corr_materiel"] for s in lignes])
    eq = np.array([s["p_win_equilibre"] for s in lignes])
    print("\n=== Synthèse ===")
    print(f"corrélation matériel/P(gain) : médiane {np.nanmedian(corr):+.3f}, "
          f"min {np.nanmin(corr):+.3f}, max {np.nanmax(corr):+.3f}")
    print(f"P(gain) à matériel équilibré : médiane {np.nanmedian(eq):.3f}, "
          f"min {np.nanmin(eq):.3f}, max {np.nanmax(eq):.3f}")
    suspects = [
        s["shard"] for s, c, e in zip(lignes, corr, eq)
        if (c < 0.2) or (not np.isnan(e) and (e < 0.30 or e > 0.70))
    ]
    if suspects:
        print(f"shards suspects ({len(suspects)}) : {', '.join(suspects[:20])}")
    else:
        print("aucun shard suspect : les étiquettes sont cohérentes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
