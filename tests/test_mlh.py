"""Vérifie la cible MLH dans les shards et la perte auxiliaire."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.model.arch_v3 import build_v3
from src.model.nets import POLICY_SIZE
from src.train.dataset import ShardDataset, collate_shard_batch
from src.train.train import compute_losses


def make_shard(path, n=16, seed=0):
    rng = np.random.default_rng(seed)
    planes = rng.integers(0, 2, size=(n, 17, 8, 8), dtype=np.uint8)
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
    plies_left = rng.integers(1, 120, size=n).astype(np.float16)
    np.savez(
        path,
        planes=planes,
        pol_idx=pol_idx,
        pol_prob=pol_prob,
        pol_len=pol_len,
        wdl=wdl,
        plies_left=plies_left,
    )
    return path


def test_mlh_target_and_auxiliary_loss(tmp_path):
    torch.manual_seed(0)
    p = make_shard(tmp_path / "shard_mlh.npz", n=32, seed=3)
    ds = ShardDataset([p])

    x, mask, idx, prob, length, wdl, plies_left, has_mlh = ds[0]
    assert plies_left.shape == ()
    assert float(has_mlh) == 1.0
    assert 0.0 < float(plies_left) < 120.0

    batch = collate_shard_batch([ds[i] for i in range(8)])
    bx, bmask, btarget, bwdl, bplies, bhas = batch
    assert bplies.shape == (8,)
    assert bhas.shape == (8,)
    assert torch.all(bhas == 1.0)

    model = build_v3("nano_ib48")
    loss, p_loss, v_loss, m_loss, policy, value, mlh = compute_losses(
        model, bx, bmask, btarget, bwdl, 1.5,
        mlh_weight=0.02, mlh_scale=100.0,
        plies_left=bplies, has_mlh=bhas,
    )
    assert mlh is not None and mlh.shape == (8, 1)
    assert math.isfinite(float(loss.detach()))
    assert math.isfinite(float(m_loss.detach())) and float(m_loss.detach()) > 0.0

    model.zero_grad()
    m_loss.backward()
    grad = model.value.fc2.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_shard_writer_stores_plies_left(tmp_path):
    from tools.lc0_to_shards import ShardWriter

    writer = ShardWriter(str(tmp_path), shard_size=2)
    planes = np.zeros((17, 8, 8), dtype=np.uint8)
    pol_idx = np.array([0, 1], dtype=np.uint16)
    pol_prob = np.array([0.5, 0.5], dtype=np.float16)
    wdl = np.array([0.2, 0.6, 0.2], dtype=np.float16)
    writer.add(planes, pol_idx, pol_prob, 2, wdl, 42.0)
    writer.add(planes, pol_idx, pol_prob, 2, wdl, 7.0)
    writer.close()

    path = tmp_path / "shard_00000.npz"
    with np.load(path) as data:
        assert "plies_left" in data
        assert np.allclose(data["plies_left"], [42.0, 7.0])
