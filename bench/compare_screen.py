"""Compare plusieurs runs de screening à partir de leurs ``history.jsonl``.

Sortie : un tableau trié par qualité, avec le coût mesuré de chaque architecture, pour
répondre à la seule question qui compte — quelle architecture donne le plus de qualité
par milliseconde sur le matériel cible.

    python bench/compare_screen.py /workspace/runs/sc_*
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def read_run(path: str) -> dict | None:
    """Lit le dernier enregistrement du history d'un run."""
    hist = os.path.join(path, "history.jsonl")
    if not os.path.isfile(hist):
        return None
    last = None
    with open(hist, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        return None
    last["run"] = os.path.basename(path.rstrip("/")).removeprefix("sc_")
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="répertoires de runs (ou globs)")
    ap.add_argument("--ms", default="", help="fichier JSON {variante: ms} pour pondérer")
    args = ap.parse_args()

    paths = []
    for pattern in args.runs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    latences = {}
    if args.ms and os.path.isfile(args.ms):
        latences = json.load(open(args.ms, encoding="utf-8"))

    rows = [r for r in (read_run(p) for p in paths) if r]
    if not rows:
        print("aucun history.jsonl trouvé")
        return 1

    rows.sort(key=lambda r: -(r.get("val", {}).get("top1") or 0))

    print(f"{'variante':16s} {'val top1':>9s} {'val pol':>8s} {'val wdl':>8s} "
          f"{'wdl acc':>8s} {'ms':>6s} {'top1/ms':>9s}")
    for r in rows:
        val = r.get("val") or {}
        top1 = val.get("top1") or 0.0
        ms = latences.get(r["run"])
        ms_txt = f"{ms:.2f}" if ms else "—"
        ratio = f"{100 * top1 / ms:.2f}" if ms else "—"
        print(
            f"{r['run']:16s} {100 * top1:8.2f}% {val.get('policy_loss', 0):8.4f} "
            f"{val.get('wdl_loss', 0):8.4f} {100 * (val.get('wdl_acc') or 0):7.2f}% "
            f"{ms_txt:>6s} {ratio:>9s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
