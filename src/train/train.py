"""Boucle d'entraînement PyTorch pour les shards DebZero.

Exemple :

    python -m src.train.train --shards /chemin/shards --variant nano \\
        --epochs 10 --batch 1024 --lr 3e-3 --amp --out runs/nano --workers 4
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.nets import POLICY_SIZE, VARIANT_SPECS, build_net
from src.train.dataset import ShardDataset, collate_shard_batch, find_shards


# --------------------------------------------------------------------------- utils
def _autocast(device: torch.device, enabled: bool):
    """Contexte autocast compatible torch 2.x (CPU/CUDA)."""
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    except (AttributeError, TypeError):  # anciennes versions
        return torch.cuda.amp.autocast(dtype=torch.float16)


def make_grad_scaler(device: torch.device, enabled: bool):
    """GradScaler compatible torch 2.x ; ``None`` si l'AMP est indisponible."""
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler(device.type, enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def load_checkpoint(path: str, map_location="cpu"):
    """``torch.load`` compatible avec ``weights_only`` récent."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _split_shards(
    shards: list[str], val_shards: int
) -> Tuple[list[str], list[str]]:
    """Sépare train/validation **par fichier**."""
    if val_shards <= 0:
        return shards, []
    if len(shards) > val_shards:
        return shards[:-val_shards], shards[-val_shards:]
    # Pas assez de shards pour isoler une validation : on réutilise les mêmes
    # fichiers. Cela évite de rendre l'entraînement impossible sur un petit jeu.
    return shards, shards[-1:]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset: ShardDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, workers),
        collate_fn=collate_shard_batch,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
        persistent_workers=workers > 0,
    )


# --------------------------------------------------------------------------- pertes
def compute_losses(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    wdl: torch.Tensor,
    wdl_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pertes policy (KL cible/probs) + WDL pondérée."""
    policy, value = model(x, mask)
    policy_loss = -(target * torch.log(policy + 1e-9)).sum(dim=1).mean()
    wdl_loss = -(wdl * torch.log(value + 1e-9)).sum(dim=1).mean()
    loss = policy_loss + wdl_weight * wdl_loss
    return loss, policy_loss, wdl_loss, policy, value


def _new_metrics() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "policy_loss": 0.0,
        "wdl_loss": 0.0,
        "top1": 0.0,
        "wdl_acc": 0.0,
        "samples": 0.0,
        "batches": 0.0,
    }


def _finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    n = max(metrics["samples"], 1.0)
    out = {
        "loss": metrics["loss"] / n,
        "policy_loss": metrics["policy_loss"] / n,
        "wdl_loss": metrics["wdl_loss"] / n,
        "top1": metrics["top1"] / n,
        "wdl_acc": metrics["wdl_acc"] / n,
        "samples": int(metrics["samples"]),
        "batches": int(metrics["batches"]),
    }
    return out


def train_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    wdl_weight: float = 1.5,
    amp: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Une époque d'entraînement. Retourne les métriques moyennées."""
    model.train()
    metrics = _new_metrics()
    amp = bool(amp and device.type == "cuda")

    for step, (x, mask, target, wdl) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            loss, p_loss, v_loss, policy, value = compute_losses(
                model, x, mask, target, wdl, wdl_weight
            )

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n = int(x.shape[0])
        metrics["loss"] += float(loss.detach()) * n
        metrics["policy_loss"] += float(p_loss.detach()) * n
        metrics["wdl_loss"] += float(v_loss.detach()) * n
        metrics["top1"] += float(
            (policy.detach().argmax(dim=1) == target.argmax(dim=1)).sum()
        )
        metrics["wdl_acc"] += float(
            (value.detach().argmax(dim=1) == wdl.argmax(dim=1)).sum()
        )
        metrics["samples"] += n
        metrics["batches"] += 1

    return _finalize_metrics(metrics)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    wdl_weight: float = 1.5,
    amp: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Évaluation sans gradient."""
    model.eval()
    metrics = _new_metrics()
    amp = bool(amp and device.type == "cuda")

    for step, (x, mask, target, wdl) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)

        with _autocast(device, amp):
            loss, p_loss, v_loss, policy, value = compute_losses(
                model, x, mask, target, wdl, wdl_weight
            )

        n = int(x.shape[0])
        metrics["loss"] += float(loss) * n
        metrics["policy_loss"] += float(p_loss) * n
        metrics["wdl_loss"] += float(v_loss) * n
        metrics["top1"] += float(
            (policy.argmax(dim=1) == target.argmax(dim=1)).sum()
        )
        metrics["wdl_acc"] += float(
            (value.argmax(dim=1) == wdl.argmax(dim=1)).sum()
        )
        metrics["samples"] += n
        metrics["batches"] += 1

    return _finalize_metrics(metrics)


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement PyTorch DebZero sur des shards .npz"
    )
    parser.add_argument("--shards", required=True, help="répertoire (ou fichier) des shards .npz")
    parser.add_argument(
        "--variant", default="base", choices=sorted(VARIANT_SPECS), help="taille du réseau"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--amp", action="store_true", help="précision mixte (CUDA uniquement)")
    parser.add_argument("--out", default=os.path.join("runs", "base"), help="répertoire de sortie")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wdl-weight", type=float, default=1.5)
    parser.add_argument("--val-shards", type=int, default=1, help="nombre de shards de validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="périphérique"
    )
    parser.add_argument("--resume", default=None, help="checkpoint last.pt à reprendre")
    parser.add_argument(
        "--smoke", action="store_true", help="20 lots seulement, pour valider la mécanique"
    )
    return parser.parse_args(argv)


def _describe(m: Dict[str, float]) -> str:
    return (
        f"loss {m['loss']:.4f} (p {m['policy_loss']:.4f}, v {m['wdl_loss']:.4f}) "
        f"top1 {m['top1']:.4f} wdl {m['wdl_acc']:.4f}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    _seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")

    shards = find_shards(args.shards)
    train_shards, val_shards = _split_shards(shards, args.val_shards)
    print(
        f"shards: {len(shards)} (train {len(train_shards)}, val {len(val_shards)}) | "
        f"variante {args.variant} | device {device} | amp {amp}",
        flush=True,
    )

    train_dataset = ShardDataset(train_shards, planes=17)
    train_loader = build_loader(
        train_dataset, args.batch, shuffle=True, workers=args.workers,
        seed=args.seed, device=device,
    )

    val_loader = None
    if val_shards:
        val_dataset = ShardDataset(val_shards, planes=17, shuffle=False)
        val_loader = build_loader(
            val_dataset, args.batch, shuffle=False, workers=args.workers,
            seed=args.seed + 1, device=device,
        )

    model = build_net(args.variant, planes=17).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    scaler = make_grad_scaler(device, amp)

    start_epoch = 0
    best_val = math.inf
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", math.inf)))
        print(
            f"repris depuis {args.resume} (époque {start_epoch}, best {best_val:.4f})",
            flush=True,
        )

    os.makedirs(args.out, exist_ok=True)
    max_batches = 20 if args.smoke else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"paramètres: {n_params:,}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device,
            wdl_weight=args.wdl_weight, amp=amp, max_batches=max_batches,
        )

        if val_loader is not None:
            val_metrics = evaluate(
                model, val_loader, device, wdl_weight=args.wdl_weight,
                amp=amp, max_batches=max_batches,
            )
            selection_loss = val_metrics["loss"]
            val_txt = f" | val {_describe(val_metrics)}"
        else:
            val_metrics = None
            selection_loss = train_metrics["loss"]
            val_txt = ""

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"époque {epoch + 1:>3}/{args.epochs} | train {_describe(train_metrics)}"
            f"{val_txt} | lr {lr:.2e} | {elapsed:.1f}s",
            flush=True,
        )

        checkpoint = {
            "variant": args.variant,
            "planes": 17,
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "val_loss": selection_loss,
            "best_val": min(best_val, selection_loss),
            "metrics": {"train": train_metrics, "val": val_metrics},
        }
        if selection_loss < best_val:
            best_val = selection_loss
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, os.path.join(args.out, "best.pt"))
            print(f"  -> nouveau best.pt (val loss {best_val:.4f})", flush=True)

        # ``last.pt`` doit permettre de reprendre : on y ajoute l'optimiseur,
        # le scheduler et l'époque.
        last = dict(checkpoint)
        last["best_val"] = best_val
        last["optimizer"] = optimizer.state_dict()
        last["scheduler"] = scheduler.state_dict()
        last["args"] = vars(args)
        torch.save(last, os.path.join(args.out, "last.pt"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
