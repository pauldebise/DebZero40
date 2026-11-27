"""Model soup entre les checkpoints best.pt et last.pt du run nano_ib48.

Interpole linéairement les poids (alpha = poids de last), exporte chaque variante en
ONNX, puis évalue sur les shards de validation : perte policy, perte WDL, perte
combinée, top-1 policy, exactitude WDL et Brier.

    python bench/soup_nano_v3.py \
        --best ~/.hermes/cache/scratch/nano_v3_ckpt/best.pt \
        --last ~/.hermes/cache/scratch/nano_v3_ckpt/last.pt \
        --shards ~/.hermes/cache/scratch/val_shards \
        --out-dir ~/.hermes/cache/scratch/nano_v3_soup
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train.export_onnx import export_onnx  # noqa: E402

POLICY_SIZE = 1858
MAX_MOVES = 64


def charger_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def faire_soupe(best_sd, last_sd, alpha):
    """alpha = poids de last ; alpha=0 -> best, alpha=1 -> last."""
    soup = {}
    for k, a in best_sd.items():
        b = last_sd[k]
        if a.is_floating_point():
            soup[k] = (1.0 - alpha) * a.float() + alpha * b.float()
        else:
            # num_batches_tracked : buffer entier, on garde la valeur de last.
            soup[k] = b.clone()
    return soup


def construire_cibles(pol_idx, pol_prob, pol_len):
    """Construit le masque de légalité et la cible policy dense."""
    n = pol_idx.shape[0]
    valid = np.arange(MAX_MOVES)[None, :] < pol_len[:, None]
    safe_idx = np.clip(pol_idx.astype(np.int64), 0, POLICY_SIZE - 1)
    rows = np.repeat(np.arange(n, dtype=np.int64), MAX_MOVES).reshape(n, MAX_MOVES)

    mask = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    mask[rows[valid], safe_idx[valid]] = 1.0

    target = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    np.add.at(target, (rows[valid], safe_idx[valid]), pol_prob[valid].astype(np.float32))
    sommes = target.sum(axis=1, keepdims=True)
    vides = sommes[:, 0] <= 0
    if vides.any():
        denom = np.clip(mask[vides].sum(axis=1, keepdims=True), 1.0, None)
        target[vides] = mask[vides] / denom
        sommes = target.sum(axis=1, keepdims=True)
    target /= np.clip(sommes, 1e-12, None)
    return mask, target


def evaluer_modele(session, shards, batch_size=4096, verbose=True):
    """Évalue un modèle ONNX sur tous les shards, un shard à la fois."""
    tot = {"n": 0, "policy": 0.0, "wdl": 0.0, "top1": 0.0,
           "wdl_acc": 0.0, "brier": 0.0}
    par_shard = {}
    for chemin in shards:
        nom = os.path.basename(chemin)
        with np.load(chemin) as d:
            planes = d["planes"]
            pol_idx = d["pol_idx"]
            pol_prob = d["pol_prob"]
            pol_len = d["pol_len"]
            wdl = d["wdl"].astype(np.float32)
        wdl = wdl / np.clip(wdl.sum(axis=1, keepdims=True), 1e-12, None)
        mask, target = construire_cibles(pol_idx, pol_prob, pol_len)
        n = planes.shape[0]
        loc = {"n": 0, "policy": 0.0, "wdl": 0.0, "top1": 0.0,
               "wdl_acc": 0.0, "brier": 0.0}
        for start in range(0, n, batch_size):
            fin = min(start + batch_size, n)
            x = planes[start:fin].transpose(0, 2, 3, 1).astype(np.float32)
            m = mask[start:fin]
            t = target[start:fin]
            w = wdl[start:fin]
            policy, wdl_pred = session.run(
                ["policy", "wdl"], {"input": x, "legal_mask": m}
            )[:2]
            policy = np.clip(policy.astype(np.float32), 1e-9, 1.0)
            wdl_pred = np.clip(wdl_pred.astype(np.float32), 1e-9, 1.0)
            b = fin - start
            loc["n"] += b
            loc["policy"] += float(-(t * np.log(policy)).sum(axis=1).sum())
            loc["wdl"] += float(-(w * np.log(wdl_pred)).sum(axis=1).sum())
            loc["top1"] += float((policy.argmax(1) == t.argmax(1)).sum())
            loc["wdl_acc"] += float((wdl_pred.argmax(1) == w.argmax(1)).sum())
            loc["brier"] += float(((wdl_pred - w) ** 2).sum(axis=1).sum())
        for k in loc:
            tot[k] += loc[k]
        par_shard[nom] = {
            "n": loc["n"],
            "policy_loss": loc["policy"] / loc["n"],
            "wdl_loss": loc["wdl"] / loc["n"],
            "top1": loc["top1"] / loc["n"],
            "wdl_acc": loc["wdl_acc"] / loc["n"],
            "brier": loc["brier"] / loc["n"],
        }
        if verbose:
            print(f"  {nom}: policy {loc['policy']/loc['n']:.4f}, "
                  f"wdl {loc['wdl']/loc['n']:.4f}, "
                  f"top1 {100*loc['top1']/loc['n']:.2f}%, "
                  f"wdl_acc {100*loc['wdl_acc']/loc['n']:.2f}%", flush=True)
    n = tot["n"]
    return {
        "n": n,
        "policy_loss": tot["policy"] / n,
        "wdl_loss": tot["wdl"] / n,
        "total": tot["policy"] / n + 1.5 * tot["wdl"] / n,
        "top1": tot["top1"] / n,
        "wdl_acc": tot["wdl_acc"] / n,
        "brier": tot["brier"] / n,
        "par_shard": par_shard,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--out-json", default="docs/mesures/soup_nano_ib48.json")
    ap.add_argument("--out-png", default="docs/mesures/soup_nano_ib48.png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.shards, "*.npz")))
    if not shards:
        raise SystemExit(f"aucun shard dans {args.shards}")

    best = charger_checkpoint(args.best)
    last = charger_checkpoint(args.last)
    best_sd = best["state_dict"]
    last_sd = last["state_dict"]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    modeles = {}
    for alpha in alphas:
        nom = f"alpha_{alpha:.2f}".replace(".", "_")
        ckpt_path = os.path.join(args.out_dir, nom + ".pt")
        onnx_path = os.path.join(args.out_dir, nom + ".onnx")
        soup = faire_soupe(best_sd, last_sd, alpha)
        torch.save({
            "variant": best.get("variant", "nano_ib48"),
            "arch": best.get("arch", "v3"),
            "planes": best.get("planes", 17),
            "state_dict": soup,
            "epoch": f"soup {alpha:.2f}",
            "metrics": {},
        }, ckpt_path)
        export_onnx(ckpt_path, onnx_path)
        modeles[nom] = onnx_path

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    resultats = {}
    for nom, chemin in modeles.items():
        print(f"\n=== {nom} ===", flush=True)
        sess = ort.InferenceSession(chemin, sess_options=so,
                                    providers=["CPUExecutionProvider"])
        resultats[nom] = evaluer_modele(sess, shards)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"alphas": alphas, "resultats": resultats}, fh,
                  ensure_ascii=False, indent=2)

    print("\n=== Récapitulatif ===")
    print(f"{'variante':<12} {'policy':>8} {'wdl':>8} {'total':>8} "
          f"{'top1':>7} {'wdl_acc':>8} {'brier':>7}")
    for alpha in alphas:
        nom = f"alpha_{alpha:.2f}".replace(".", "_")
        r = resultats[nom]
        print(f"{nom:<12} {r['policy_loss']:8.4f} {r['wdl_loss']:8.4f} "
              f"{r['total']:8.4f} {100*r['top1']:6.2f}% {100*r['wdl_acc']:7.2f}% "
              f"{r['brier']:7.4f}")

    # Figure : pertes + exactitudes en fonction de alpha.
    xs = alphas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    fig.suptitle("Model soup nano_ib48 : interpolation best (ép. 15) → last (ép. 16)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    ax = axes[0]
    ax.plot(xs, [resultats[f"alpha_{a:.2f}".replace(".", "_")]["policy_loss"] for a in xs],
            "o-", color="#1f6feb", label="perte politique")
    ax.plot(xs, [resultats[f"alpha_{a:.2f}".replace(".", "_")]["wdl_loss"] for a in xs],
            "o-", color="#d1541f", label="perte WDL")
    ax.plot(xs, [resultats[f"alpha_{a:.2f}".replace(".", "_")]["total"] for a in xs],
            "o--", color="#8250df", label="perte totale (p + 1,5·wdl)")
    ax.set_xlabel("alpha (poids de last)")
    ax.set_ylabel("perte de validation")
    ax.set_title("Pertes", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(xs, [100 * resultats[f"alpha_{a:.2f}".replace(".", "_")]["top1"] for a in xs],
            "o-", color="#1a7f37", label="top-1 politique")
    ax.plot(xs, [100 * resultats[f"alpha_{a:.2f}".replace(".", "_")]["wdl_acc"] for a in xs],
            "o-", color="#bf8700", label="exactitude WDL")
    ax.set_xlabel("alpha (poids de last)")
    ax.set_ylabel("pourcentage")
    ax.set_title("Exactitudes", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"\nécrit : {args.out_json} et {args.out_png}")


if __name__ == "__main__":
    main()
