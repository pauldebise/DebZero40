"""Quantifie un réseau ONNX et mesure le vrai gain : latence et qualité.

Trois variantes comparées sur les mêmes positions :
  - fp32 d'origine
  - quantization dynamique (poids int8, activations calculées à la volée) — ne touche
    que les couches denses, donc peu d'effet sur un réseau convolutif
  - quantization statique QDQ (poids et activations int8, calibrage sur des positions
    réelles) — la seule qui accélère vraiment les convolutions

La latence retenue est le minimum sur plusieurs passes : c'est le régime du bot, qui
enchaîne les positions sans pause.

    python bench/quantize_net.py --onnx net.onnx --calib positions.npz --out-dir /tmp/q
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import onnxruntime as ort


def load_positions(path: str, limit: int | None = None):
    with np.load(path) as d:
        planes = d["planes"]
        pol_idx = d["pol_idx"]
        pol_prob = d["pol_prob"]
        pol_len = d["pol_len"]
    if limit:
        planes, pol_idx, pol_prob, pol_len = (
            planes[:limit], pol_idx[:limit], pol_prob[:limit], pol_len[:limit]
        )
    x = planes.astype(np.float32).transpose(0, 2, 3, 1)
    mask = np.zeros((x.shape[0], 1858), dtype=np.float32)
    for r in range(x.shape[0]):
        mask[r, pol_idx[r, : int(pol_len[r])]] = 1.0
    best = np.array([pol_idx[r, int(np.argmax(pol_prob[r, : int(pol_len[r])]))]
                     for r in range(x.shape[0])], dtype=np.int64)
    return x, mask, best


def session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def latency(sess: ort.InferenceSession, x: np.ndarray, mask: np.ndarray, passes: int = 60) -> float:
    feed = {sess.get_inputs()[0].name: x[:1], sess.get_inputs()[1].name: mask[:1]}
    for _ in range(10):
        sess.run(None, feed)
    best = float("inf")
    for _ in range(passes):
        t0 = time.perf_counter()
        sess.run(None, feed)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def quality(sess: ort.InferenceSession, x: np.ndarray, mask: np.ndarray,
            best: np.ndarray, chunk: int = 512):
    top1 = 0
    for i in range(0, x.shape[0], chunk):
        out = sess.run(None, {sess.get_inputs()[0].name: x[i:i + chunk],
                              sess.get_inputs()[1].name: mask[i:i + chunk]})
        policy = np.asarray(out[0])
        top1 += int((policy.argmax(1) == best[i:i + chunk]).sum())
    return 100.0 * top1 / x.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True, help="npz de positions pour le calibrage")
    ap.add_argument("--out-dir", default="/tmp/quant")
    ap.add_argument("--calib-positions", type=int, default=500)
    ap.add_argument("--eval-positions", type=int, default=5000)
    args = ap.parse_args()

    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    x, mask, best = load_positions(args.calib, limit=max(args.calib_positions, args.eval_positions))

    names = [i.name for i in session(args.onnx).get_inputs()]

    class Feed(CalibrationDataReader):
        def __init__(self, xa, ma):
            self.data = [{names[0]: xa[i:i + 1], names[1]: ma[i:i + 1]}
                         for i in range(len(xa))]
            self.i = 0

        def get_next(self):  # type: ignore[override]
            if self.i >= len(self.data):
                return None
            d = self.data[self.i]
            self.i += 1
            return d

        def rewind(self):
            self.i = 0

    dyn = os.path.join(args.out_dir, "dynamic.onnx")
    sta = os.path.join(args.out_dir, "static.onnx")

    quantize_dynamic(args.onnx, dyn, weight_type=QuantType.QInt8)
    quantize_static(
        args.onnx, sta,
        Feed(x[: args.calib_positions], mask[: args.calib_positions]),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )

    x = x[: args.eval_positions]
    mask = mask[: args.eval_positions]
    best = best[: args.eval_positions]

    print(f"{'variante':10s} {'taille':>9s} {'ms':>6s} {'top-1':>7s}")
    for label, path in (("fp32", args.onnx), ("dynamique", dyn), ("statique", sta)):
        s = session(path)
        ms = latency(s, x, mask)
        acc = quality(s, x, mask, best)
        ko = os.path.getsize(path) / 1024
        print(f"{label:10s} {ko:8.0f}Ko {ms:6.2f} {acc:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
