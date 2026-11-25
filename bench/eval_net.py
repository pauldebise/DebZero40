"""Évalue un réseau entraîné sur des positions LC0 non vues, et mesure sa vitesse.

Mesures : top-1 de la policy, exactitude WDL, temps d'inférence par position.
Sert de compte rendu objectif après un run d'entraînement.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from src.model.nets import build_net


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    variant = ckpt.get("variant") or ckpt.get("config", {}).get("variant", "base")
    model = build_net(variant)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model")
    model.load_state_dict(state)
    model.eval()
    return model, variant


def evaluate(model, files, max_positions=20000, batch=512):
    n = 0
    top1 = 0
    wdl_ok = 0
    total = 0
    t_inf = 0.0

    for path in files:
        with np.load(path) as d:
            planes = d["planes"]
            pol_idx = d["pol_idx"]
            pol_prob = d["pol_prob"]
            pol_len = d["pol_len"]
            wdl = d["wdl"]
            count = min(len(planes), max_positions - n)
            if count <= 0:
                break
            for start in range(0, count, batch):
                sl = slice(start, min(start + batch, count))
                x = torch.from_numpy(np.ascontiguousarray(planes[sl])).float()
                # Les shards stockent (N, plans, 8, 8) ; le réseau attend NHWC (N, 8, 8, plans).
                x = x.permute(0, 2, 3, 1)
                idx = torch.from_numpy(np.ascontiguousarray(pol_idx[sl])).long()
                prob = torch.from_numpy(np.ascontiguousarray(pol_prob[sl])).float()
                ln = torch.from_numpy(np.ascontiguousarray(pol_len[sl])).long()
                y = torch.from_numpy(np.ascontiguousarray(wdl[sl])).float()

                mask = torch.zeros((x.shape[0], 1858), dtype=torch.float32)
                for r in range(x.shape[0]):
                    mask[r, idx[r, : int(ln[r])]] = 1.0

                t0 = time.perf_counter()
                with torch.no_grad():
                    p, v = model(x, mask)
                t_inf += time.perf_counter() - t0

                # top-1 : le coup le plus probable doit être le plus probable de la cible
                target_best = torch.zeros(x.shape[0], dtype=torch.long)
                for r in range(x.shape[0]):
                    row = prob[r, : int(ln[r])]
                    target_best[r] = idx[r, int(row.argmax())]
                top1 += int((p.argmax(1) == target_best).sum())
                wdl_ok += int((v.argmax(1) == y.argmax(1)).sum())
                total += x.shape[0]
                n += x.shape[0]
        if n >= max_positions:
            break

    return {
        "positions": total,
        "top1": top1 / max(total, 1),
        "wdl_acc": wdl_ok / max(total, 1),
        "ms_per_pos": 1000.0 * t_inf / max(total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--max-positions", type=int, default=20000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.shards, "*.npz")))
    model, variant = load_checkpoint(args.checkpoint)
    nparams = sum(p.numel() for p in model.parameters())
    res = evaluate(model, files, args.max_positions)
    print(f"{os.path.basename(args.checkpoint)} ({variant}, {nparams/1e6:.2f} M params)")
    print(f"  positions évaluées : {res['positions']}")
    print(f"  top-1 policy       : {res['top1']*100:.2f} %")
    print(f"  exactitude WDL     : {res['wdl_acc']*100:.2f} %")
    print(f"  inférence (CPU)    : {res['ms_per_pos']:.2f} ms/position")


if __name__ == "__main__":
    main()
