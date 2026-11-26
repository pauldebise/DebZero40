"""Mesure paramètres, opérations et latence des architectures candidates.

Trois chiffres par modèle, tous mesurés et non estimés :

* **paramètres** — taille du fichier ;
* **opérations** — MAC comptés par crochets sur les couches réelles (1 MAC = 1 op) ;
* **latence** — millisecondes par position sur un seul cœur, via onnxruntime, ce qui est
  le chemin réel du moteur.

Le « nps » affiché est le débit attendu du moteur complet : latence du réseau plus
l'overhead du MCTS Python, mesuré à 0,55 ms par nœud sur le VPS.

    python bench/bench_arch.py --models v3:nano v3:nano_dense v2:nano v2:base
"""
from __future__ import annotations

import argparse
import inspect
import os
import statistics
import tempfile
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.model.arch_v3 import PRESETS, DebZeroV3, V3Config, build_v3
from src.model.nets import build_net

OVERHEAD_MS = 0.55


def count_macs(model: nn.Module, planes: int = 17) -> int:
    """Compte les MAC d'une passe avant en interceptant chaque couche."""
    total = 0
    handles = []

    def conv_hook_clean(module: nn.Conv2d, inp, out):
        nonlocal total
        out_elems = out.shape[0] * out.shape[2] * out.shape[3]
        total += out_elems * module.out_channels * (module.in_channels // module.groups) * \
            module.kernel_size[0] * module.kernel_size[1]

    def linear_hook(module: nn.Linear, inp, out):
        nonlocal total
        total += out.shape[0] * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook_clean))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 8, 8, planes))
    for h in handles:
        h.remove()
    return total


def export_and_time(model: nn.Module, planes: int, repeats: int = 60) -> float:
    """Exporte en ONNX et renvoie la latence **minimale** par position, un seul thread.

    Le minimum, et non la médiane : le VPS héberge le bot, une mesure médiane capture ses
    pics de charge. Le minimum approche le coût réel du réseau sur un cœur au repos.
    """
    import contextlib
    import io

    import onnxruntime as ort

    model.eval()
    dummy_x = torch.zeros(1, 8, 8, planes)
    dummy_mask = torch.ones(1, 1858)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m.onnx")
    kwargs = {"dynamo": True} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(
            model, (dummy_x, dummy_mask), path, opset_version=17, do_constant_folding=True,
            input_names=["input", "legal_mask"],
            output_names=["policy", "wdl", "mlh"],
            dynamic_axes={"input": {0: "batch"}, "legal_mask": {0: "batch"},
                          "policy": {0: "batch"}, "wdl": {0: "batch"}, "mlh": {0: "batch"}},
            **kwargs,
        )
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    feed = {
        "input": np.zeros((1, 8, 8, planes), dtype=np.float32),
        "legal_mask": np.ones((1, 1858), dtype=np.float32),
    }
    for _ in range(5):
        sess.run(None, feed)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)


def build(spec: str):
    """``v3:nano`` ou ``v2:base`` -> (modèle, plans)."""
    family, _, name = spec.partition(":")
    if family == "v3":
        overrides = {}
        if name.endswith("_dense"):
            overrides["head"] = "dense"
            name = name[: -len("_dense")]
        elif name.endswith("_hybrid"):
            overrides["head"] = "hybrid"
            name = name[: -len("_hybrid")]
        return build_v3(name, **overrides), 17
    if family == "v2":
        return build_net(name), 17
    raise ValueError(spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["v3:nano", "v3:nano_dense", "v3:nano_hybrid", "v2:nano", "v2:base"])
    ap.add_argument("--budget", type=float, default=5.0, help="budget d'opérations visé, en M")
    args = ap.parse_args()

    print(f"{'modèle':18s} {'params':>9s} {'M op':>7s} {'ms/pos':>8s} {'nps':>6s} {'verdict':>10s}")
    rows: List[tuple] = []
    for spec in args.models:
        model, planes = build(spec)
        params = sum(p.numel() for p in model.parameters())
        macs = count_macs(model, planes)
        ms = export_and_time(model, planes)
        nps = 1000.0 / (ms + OVERHEAD_MS)
        verdict = "ok" if macs / 1e6 <= args.budget * 1.05 else "hors budget"
        rows.append((spec, params, macs, ms, nps))
        print(f"{spec:18s} {params/1e6:8.2f}M {macs/1e6:7.2f} {ms:8.2f} {nps:6.0f} {verdict:>10s}")

    print(f"\nbudget visé : {args.budget:.1f} M op/position | overhead MCTS retenu : {OVERHEAD_MS:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
