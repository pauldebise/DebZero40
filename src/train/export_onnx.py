"""Export d'un checkpoint PyTorch vers ONNX (opset 17) et vérification onnxruntime.

Exemple :

    python -m src.train.export_onnx --checkpoint runs/nano/best.pt \\
        --out nets/nano_fp32.onnx

Le modèle exporté a deux entrées nommées ``input`` ``(batch, 8, 8, 17)`` et
``legal_mask`` ``(batch, 1858)``, et deux sorties ``policy`` ``(batch, 1858)`` et
``wdl`` ``(batch, 3)``.  Le batch est dynamique.
"""
from __future__ import annotations

import argparse
import inspect
import os

import numpy as np
import torch

from src.model.nets import POLICY_SIZE, build_net
from src.train.train import load_checkpoint


def _checkpoint_variant(ckpt) -> tuple[str, str, int]:
    """Retrouve la famille, la variante et le nombre de plans depuis le checkpoint."""
    variant = ckpt.get("variant")
    arch = ckpt.get("arch", "v2")
    planes = ckpt.get("planes", 17)
    if variant is None:
        config = ckpt.get("config") or ckpt.get("args") or {}
        variant = config.get("variant", "base")
        arch = config.get("arch", arch)
        planes = config.get("planes", planes)
    return str(arch), str(variant), int(planes)


def export_onnx(checkpoint_path: str, out_path: str, opset: int = 17) -> str:
    """Charge ``best.pt``, reconstruit le réseau et exporte en ONNX."""
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    arch, variant, planes = _checkpoint_variant(ckpt)
    if arch == "v3":
        from src.model.arch_v3 import build_v3

        model = build_v3(variant)
        output_names = ["policy", "wdl", "mlh"]
    else:
        model = build_net(variant, planes=planes)
        output_names = ["policy", "wdl"]
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    dummy_x = torch.randn(2, 8, 8, planes, dtype=torch.float32)
    dummy_mask = torch.zeros(2, POLICY_SIZE, dtype=torch.float32)
    legal = torch.randint(0, POLICY_SIZE, (2, 24))
    dummy_mask.scatter_(1, legal, 1.0)

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    dynamic_axes = {
        "input": {0: "batch"},
        "legal_mask": {0: "batch"},
        "policy": {0: "batch"},
        "wdl": {0: "batch"},
    }
    if arch == "v3":
        dynamic_axes["mlh"] = {0: "batch"}
    kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # Le nouvel exportateur dynamo ne gère pas encore tous les cas opset 17 ;
        # l'exporteur TorchScript est stable pour ce réseau.
        kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_x, dummy_mask),
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input", "legal_mask"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **kwargs,
        )

    size = os.path.getsize(out_path)
    print(f"ONNX exporté : {out_path} ({size / (1024 * 1024):.2f} Mio, opset {opset})")

    # ------------------------------------------------------------- vérification
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise RuntimeError(
            "onnxruntime est requis pour vérifier l'export (pip install onnxruntime)"
        ) from exc

    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in session.get_inputs()}
    outputs = {o.name: o for o in session.get_outputs()}
    assert set(inputs) == {"input", "legal_mask"}, f"entrées inattendues : {sorted(inputs)}"
    attendues = {"policy", "wdl"} | ({"mlh"} if arch == "v3" else set())
    assert set(outputs) == attendues, f"sorties inattendues : {sorted(outputs)}"

    # Batch différent de celui de l'export : vérifie le caractère dynamique.
    rng = np.random.default_rng(1234)
    batch = 5
    x = rng.standard_normal((batch, 8, 8, planes), dtype=np.float32)
    mask = np.zeros((batch, POLICY_SIZE), dtype=np.float32)
    for row in range(batch):
        mask[row, rng.choice(POLICY_SIZE, size=24, replace=False)] = 1.0

    policy, wdl = session.run(
        ["policy", "wdl"], {"input": x, "legal_mask": mask}
    )
    legal_mass = (policy * mask).sum(axis=1)
    if not np.allclose(legal_mass, 1.0, atol=1e-4):
        raise AssertionError(
            f"masse légale de la policy = {legal_mass}, attendu 1.0 à 1e-4 près"
        )
    if not np.allclose(policy.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("la policy exportée ne somme pas à 1")
    if not np.allclose(wdl.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("la sortie wdl exportée ne somme pas à 1")

    print(
        f"vérification onnxruntime : batch {batch}, "
        f"masse légale {legal_mass.min():.8f}..{legal_mass.max():.8f}, OK"
    )
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporte un checkpoint DebZero en ONNX")
    parser.add_argument("--checkpoint", required=True, help="chemin de best.pt / last.pt")
    parser.add_argument("--out", required=True, help="fichier ONNX de sortie")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    export_onnx(args.checkpoint, args.out, opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
