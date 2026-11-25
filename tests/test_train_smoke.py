"""Test fumée de la boucle d'entraînement : deux lots, perte finie, poids modifiés."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.nets import POLICY_SIZE, build_net
from src.train.dataset import ShardDataset, collate_shard_batch
from src.train.train import train_epoch


def make_shard(path, n=64, planes=17, seed=0):
    rng = np.random.default_rng(seed)
    boards = rng.integers(0, 2, size=(n, planes, 8, 8), dtype=np.uint8)

    pol_len = rng.integers(1, 65, size=n).astype(np.uint8)
    pol_idx = np.zeros((n, 64), dtype=np.uint16)
    pol_prob = np.zeros((n, 64), dtype=np.float16)
    for i in range(n):
        k = int(pol_len[i])
        idx = rng.choice(POLICY_SIZE, size=k, replace=False).astype(np.uint16)
        prob = rng.random(k).astype(np.float32)
        prob /= prob.sum()
        pol_idx[i, :k] = idx
        pol_prob[i, :k] = prob.astype(np.float16)

    wdl = np.zeros((n, 3), dtype=np.float16)
    wdl[np.arange(n), rng.integers(0, 3, size=n)] = 1.0
    np.savez(
        path,
        planes=boards,
        pol_idx=pol_idx,
        pol_prob=pol_prob,
        pol_len=pol_len,
        wdl=wdl,
    )
    return path


def test_two_batches_loss_finite_and_weights_change(tmp_path):
    torch.manual_seed(0)
    p1 = make_shard(tmp_path / "shard_000.npz", n=64, seed=5)
    p2 = make_shard(tmp_path / "shard_001.npz", n=64, seed=6)

    ds = ShardDataset([p1, p2])
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_shard_batch,
    )

    model = build_net("nano")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    metrics = train_epoch(
        model,
        loader,
        optimizer,
        scaler=None,
        device=torch.device("cpu"),
        wdl_weight=1.5,
        amp=False,
        max_batches=2,
    )

    assert metrics["batches"] == 2
    assert metrics["samples"] == 16
    for key in ("loss", "policy_loss", "wdl_loss", "top1", "wdl_acc"):
        assert math.isfinite(metrics[key]), f"{key} non fini : {metrics[key]}"

    changed = [
        name
        for name, p in model.named_parameters()
        if not torch.equal(p.detach(), before[name])
    ]
    assert changed, "aucun poids n'a changé après deux lots"
    print(f"  loss={metrics['loss']:.4f}, {len(changed)} tenseurs modifiés")
