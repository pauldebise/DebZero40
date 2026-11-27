"""Tests du zoo de réseaux PyTorch et du dataset de shards."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.model.nets import POLICY_SIZE, VARIANT_SPECS, build_net, count_parameters
from src.train.dataset import ShardDataset, collate_shard_batch, find_shards


# --------------------------------------------------------------------------- helpers
def make_shard(path, n=64, planes=17, seed=0):
    """Fabrique un shard .npz minimal mais réaliste."""
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


def random_mask(batch, legal_per_row=24, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((batch, POLICY_SIZE), dtype=np.float32)
    for row in range(batch):
        mask[row, rng.choice(POLICY_SIZE, size=legal_per_row, replace=False)] = 1.0
    return torch.from_numpy(mask)


# --------------------------------------------------------------------------- réseau
def test_variants_shapes_and_parameter_growth():
    counts = []
    for variant in ("nano", "small", "base", "large"):
        assert variant in VARIANT_SPECS
        model = build_net(variant)
        model.eval()
        n_params = count_parameters(model)
        assert n_params > 0
        counts.append(n_params)

        x = torch.randn(2, 8, 8, 17)
        mask = random_mask(2, seed=len(counts))
        with torch.no_grad():
            policy, wdl = model(x, mask)

        assert policy.shape == (2, POLICY_SIZE)
        assert wdl.shape == (2, 3)
        assert policy.dtype == torch.float32
        assert wdl.dtype == torch.float32
        assert torch.allclose(policy.sum(dim=1), torch.ones(2), atol=1e-5)
        assert torch.allclose(wdl.sum(dim=1), torch.ones(2), atol=1e-5)

    assert all(b > a for a, b in zip(counts, counts[1:])), counts
    print(f"  paramètres par variante : {counts}")


def test_legal_mask_really_zeroes_illegal_moves():
    model = build_net("base").eval()
    x = torch.randn(3, 8, 8, 17)
    mask = random_mask(3, legal_per_row=12, seed=7)

    with torch.no_grad():
        policy, _ = model(x, mask)

    legal_mass = (policy * mask).sum(dim=1)
    assert torch.allclose(legal_mass, torch.ones(3), atol=1e-6)

    illegal = mask == 0
    assert torch.count_nonzero(policy[illegal]) == 0, "un coup illégal a reçu une probabilité"

    # Sur chaque ligne le nombre de probabilités non nulles ne dépasse pas le
    # nombre de coups légaux.
    nonzero = (policy > 0).sum(dim=1)
    assert torch.all(nonzero <= mask.sum(dim=1))


def test_nhwc_input_is_not_accidentally_invariant():
    """Une permutation spatiale cohérente de l'entrée doit changer la sortie."""
    model = build_net("nano").eval()
    x = torch.randn(2, 8, 8, 17)
    mask = torch.ones(2, POLICY_SIZE)

    with torch.no_grad():
        policy_a, wdl_a = model(x, mask)
        # Transposition hauteur/largeur : le plateau « vu » change, la sortie aussi.
        policy_b, wdl_b = model(x.transpose(1, 2), mask)

    assert policy_a.shape == (2, POLICY_SIZE)
    assert not torch.allclose(policy_a, policy_b, atol=1e-6)
    assert not torch.allclose(wdl_a, wdl_b, atol=1e-6)

    # Le réseau consomme bien du NHWC : lui donner du NCHW doit échouer.
    with pytest.raises(RuntimeError):
        model(x.permute(0, 3, 1, 2), mask)


# --------------------------------------------------------------------------- dataset
def test_dataset_shapes_targets_and_mask(tmp_path):
    p1 = make_shard(tmp_path / "shard_000.npz", n=64, seed=1)
    p2 = make_shard(tmp_path / "shard_001.npz", n=64, seed=2)

    shards = find_shards(tmp_path)
    assert shards == [str(p1), str(p2)]

    ds = ShardDataset([p1, p2])
    assert len(ds) == 128

    x, mask, idx, prob, length, wdl, plies_left, has_mlh = ds[0]
    assert x.shape == (8, 8, 17)
    assert x.dtype == torch.float32
    assert mask.shape == (POLICY_SIZE,)
    assert mask.dtype == torch.float32
    assert wdl.shape == (3,)
    assert wdl.dtype == torch.float32
    assert plies_left.shape == ()
    assert float(has_mlh) == 0.0  # shard historique sans cible MLH

    # x est bien la transposition NHWC des plans NCHW du shard.
    with np.load(p1) as data:
        expected = np.transpose(data["planes"][0], (1, 2, 0)).astype(np.float32)
    assert np.array_equal(x.numpy(), expected)

    n_legal = int(length)
    assert n_legal > 0
    assert int(mask.sum()) == n_legal
    assert torch.all(mask[idx[:n_legal].long()] == 1.0)

    batch = collate_shard_batch([ds[i] for i in range(8)])
    bx, bmask, btarget, bwdl, bplies, bhas = batch
    assert bx.shape == (8, 8, 8, 17)
    assert bmask.shape == (8, POLICY_SIZE)
    assert btarget.shape == (8, POLICY_SIZE)
    assert btarget.dtype == torch.float32
    assert bwdl.shape == (8, 3)
    assert bplies.shape == (8,)
    assert bhas.shape == (8,)
    assert torch.all(bhas == 0)

    assert torch.allclose(btarget.sum(dim=1), torch.ones(8), atol=1e-5)
    assert torch.all(btarget >= 0)
    # Toute masse de la cible se trouve sur un coup marqué légal.
    assert torch.all((btarget > 0) <= (bmask == 1))
    assert torch.allclose(bwdl.sum(dim=1), torch.ones(8), atol=1e-5)


def test_dataset_shuffled_order_keeps_all_positions(tmp_path):
    p1 = make_shard(tmp_path / "a.npz", n=16, seed=3)
    p2 = make_shard(tmp_path / "b.npz", n=16, seed=4)
    ds = ShardDataset([p1, p2], shuffle=True, seed=11)
    assert len(ds) == 32
    # Aucun index hors bornes et toutes les positions restent accessibles.
    seen = {tuple(ds[i][0].flatten().tolist()) for i in range(len(ds))}
    assert len(seen) > 1
    for i in range(len(ds)):
        assert math.isfinite(float(ds[i][0].sum()))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
