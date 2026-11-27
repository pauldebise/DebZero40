"""Audit de calibration de la tête WDL sur les shards de validation.

Charge les shards, fait tourner le modèle ONNX par lots, puis compare les probabilités
prédites aux cibles WDL de LC0. Sort un JSON et une figure de fiabilité.

    python bench/audit_wdl_calibration.py \
        --shards ~/.hermes/cache/scratch/val_shards \
        --model nets/v3/nano_ib48.onnx \
        --out-json docs/mesures/calibration_wdl_nano_ib48.json \
        --out-png docs/mesures/calibration_wdl_nano_ib48.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort


CLASSES = ("victoire", "nulle", "défaite")
COULEURS = ("#1f6feb", "#8250df", "#d1541f")


def entropie(p):
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)


def ece_binaire(pred, cible, n_bins=10):
    """ECE d'une probabilité binaire : bins de largeur fixe sur [0, 1]."""
    bornes = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, bornes) - 1, 0, n_bins - 1)
    ece = 0.0
    detail = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        mp = float(pred[m].mean())
        mt = float(cible[m].mean())
        ece += n / len(pred) * abs(mp - mt)
        detail.append({"bin": [float(bornes[b]), float(bornes[b + 1])], "n": n,
                       "predit": mp, "observe": mt})
    return float(ece), detail


def reliability(pred, cible, n_bins=10):
    bornes = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, bornes) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        xs.append(float(pred[m].mean()))
        ys.append(float(cible[m].mean()))
        ns.append(n)
    return xs, ys, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-json", default="docs/mesures/calibration_wdl_nano_ib48.json")
    ap.add_argument("--out-png", default="docs/mesures/calibration_wdl_nano_ib48.png")
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards, "*.npz")))
    if not shards:
        raise SystemExit(f"aucun shard dans {args.shards}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=["CPUExecutionProvider"])
    mask = np.ones((args.batch, 1858), dtype=np.float32)

    all_t, all_p, all_n, all_ce, all_shard = [], [], [], [], []
    par_shard = {}
    for chemin in shards:
        nom = os.path.basename(chemin)
        with np.load(chemin) as d:
            planes = d["planes"]
            wdl = d["wdl"].astype(np.float32)
        # Les cibles sont normalisées dans les shards ; on renormalise par sécurité.
        wdl = wdl / np.clip(wdl.sum(axis=1, keepdims=True), 1e-12, None)
        n = planes.shape[0]
        preds = []
        for start in range(0, n, args.batch):
            bloc = planes[start:start + args.batch].transpose(0, 2, 3, 1).astype(np.float32)
            b = bloc.shape[0]
            sortie = sess.run(None, {"input": bloc, "legal_mask": mask[:b]})[1]
            preds.append(sortie.astype(np.float32))
        pred = np.concatenate(preds, axis=0)
        pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-12, None)
        ce = -(wdl * np.log(np.clip(pred, 1e-9, 1.0))).sum(axis=1)
        pieces = planes[:, :12].sum(axis=(1, 2, 3)).astype(np.int16)
        all_t.append(wdl)
        all_p.append(pred)
        all_n.append(pieces)
        all_ce.append(ce)
        all_shard.append(np.full(n, nom, dtype=object))
        par_shard[nom] = {
            "n": int(n),
            "ce": float(ce.mean()),
            "cible_nulle": float(wdl[:, 1].mean()),
            "predite_nulle": float(pred[:, 1].mean()),
            "exactitude": float((pred.argmax(1) == wdl.argmax(1)).mean()),
            "brier": float(((pred - wdl) ** 2).sum(axis=1).mean()),
        }
        print(f"{nom}: {n} positions, CE {ce.mean():.4f}, "
              f"cible nulle {wdl[:, 1].mean():.3f}, prédite {pred[:, 1].mean():.3f}", flush=True)

    t = np.concatenate(all_t)
    p = np.concatenate(all_p)
    pieces = np.concatenate(all_n)
    ce = np.concatenate(all_ce)

    res = {
        "n": int(len(t)),
        "shards": par_shard,
        "cible_moyenne": {c: float(t[:, i].mean()) for i, c in enumerate(CLASSES)},
        "predite_moyenne": {c: float(p[:, i].mean()) for i, c in enumerate(CLASSES)},
        "entropie_cible": float(entropie(t).mean()),
        "entropie_predite": float(entropie(p).mean()),
        "ce": {"moyenne": float(ce.mean()), "ecart_type": float(ce.std()),
               "p50": float(np.percentile(ce, 50)), "p90": float(np.percentile(ce, 90)),
               "p95": float(np.percentile(ce, 95)), "p99": float(np.percentile(ce, 99)),
               "p999": float(np.percentile(ce, 99.9)), "max": float(ce.max()),
               "part_au_dessus_ln3": float((ce > np.log(3)).mean()),
               "part_au_dessus_2": float((ce > 2.0).mean()),
               "part_au_dessus_3": float((ce > 3.0).mean())},
        "exactitude": float((p.argmax(1) == t.argmax(1)).mean()),
        "brier": float(((p - t) ** 2).sum(axis=1).mean()),
        "ece": {},
        "calibration": {},
    }
    for i, c in enumerate(CLASSES):
        e, detail = ece_binaire(p[:, i], t[:, i])
        res["ece"][c] = e
        res["calibration"][c] = detail
        xs, ys, ns = reliability(p[:, i], t[:, i])
        res["calibration"][c + "_reliability"] = {"predit": xs, "observe": ys, "n": ns}

    # CE par nombre de pièces (phase de partie).
    tranches = [(2, 8), (9, 16), (17, 24), (25, 32)]
    par_phase = {}
    for lo, hi in tranches:
        m = (pieces >= lo) & (pieces <= hi)
        if m.sum() == 0:
            continue
        par_phase[f"{lo}-{hi}"] = {
            "n": int(m.sum()),
            "ce": float(ce[m].mean()),
            "cible_nulle": float(t[m, 1].mean()),
            "predite_nulle": float(p[m, 1].mean()),
            "exactitude": float((p[m].argmax(1) == t[m].argmax(1)).mean()),
        }
    res["par_phase"] = par_phase
    res["correlation_ce_pieces"] = float(np.corrcoef(ce, pieces)[0, 1])
    res["correlation_ce_entropie_cible"] = float(np.corrcoef(ce, entropie(t))[0, 1])
    res["correlation_ce_entropie_predite"] = float(np.corrcoef(ce, entropie(p))[0, 1])

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2)[:4000])

    # Figure : fiabilité + distribution de la CE + par shard + par phase.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)
    fig.suptitle("Calibration WDL du nano_ib48 sur 250 000 positions de validation",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")

    ax = axes[0][0]
    ax.plot([0, 1], [0, 1], "--", color="#8c959f", lw=1, label="calibration parfaite")
    for i, c in enumerate(CLASSES):
        rel = res["calibration"][c + "_reliability"]
        ax.plot(rel["predit"], rel["observe"], "o-", color=COULEURS[i], lw=1.6, ms=4,
                label=f"{c} (ECE {res['ece'][c]:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("probabilité prédite")
    ax.set_ylabel("fréquence observée")
    ax.set_title("Diagramme de fiabilité", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0][1]
    ax.hist(ce, bins=80, color="#1f6feb", alpha=0.85)
    ax.axvline(ce.mean(), color="#d1541f", lw=1.5, label=f"moyenne {ce.mean():.3f}")
    ax.axvline(np.log(3), color="#8c959f", ls="--", lw=1.2, label="hasard ln 3")
    ax.set_xlabel("entropie croisée par position")
    ax.set_ylabel("nombre de positions")
    ax.set_title("Distribution de la perte WDL", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.2, ls=":")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][0]
    noms = list(par_shard)
    x = np.arange(len(noms))
    ax.bar(x - 0.18, [par_shard[n]["cible_nulle"] for n in noms], width=0.36,
           color="#8250df", label="cible nulle")
    ax.bar(x + 0.18, [par_shard[n]["predite_nulle"] for n in noms], width=0.36,
           color="#d1541f", label="prédite nulle")
    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Probabilité de nulle, par shard", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.2, axis="y", ls=":")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    labels = list(par_phase)
    vals = [par_phase[k]["ce"] for k in labels]
    ax.bar(labels, vals, color="#1a7f37")
    for i, k in enumerate(labels):
        ax.text(i, vals[i] + 0.02, f"{vals[i]:.3f}", ha="center", fontsize=8)
    ax.set_xlabel("nombre de pièces sur l'échiquier")
    ax.set_ylabel("perte WDL moyenne")
    ax.set_title("Perte WDL par phase", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.2, axis="y", ls=":")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"écrit : {args.out_json} et {args.out_png}")


if __name__ == "__main__":
    main()
