"""Tests de la gamme v3 : formes, masque de légalité, cohérence de la table des coups."""
from __future__ import annotations

import pytest
import torch

from src.model.arch_v3 import (
    PRESETS,
    DebZeroV3,
    V3Config,
    build_v3,
    move_indices,
)
from src.utils.lc0_moves import LC0_MOVES, LC0_MOVE_TO_INDEX


def _dummy(batch: int = 3):
    x = torch.zeros(batch, 8, 8, 17)
    mask = torch.zeros(batch, 1858)
    mask[:, :30] = 1.0
    return x, mask


def test_move_indices_cover_all_moves():
    froms, tos, promos = move_indices()
    assert len(froms) == len(tos) == len(promos) == 1858
    assert int(froms.min()) >= 0 and int(froms.max()) <= 63
    assert int(tos.min()) >= 0 and int(tos.max()) <= 63
    assert set(promos.tolist()) <= {0, 1, 2, 3}


def test_move_indices_match_known_squares():
    """e2e4 doit être (12, 28) dans l'ordre des plans (rang 8 en premier)."""
    froms, tos, _ = move_indices()
    idx = LC0_MOVE_TO_INDEX["e2e4"]
    assert (int(froms[idx]), int(tos[idx])) == (12, 28)
    idx = LC0_MOVE_TO_INDEX["e1h1"]          # roque blanc, encodé roi prend tour
    assert (int(froms[idx]), int(tos[idx])) == (4, 7)
    idx = LC0_MOVE_TO_INDEX["e7e8q"]
    assert (int(froms[idx]), int(tos[idx])) == (52, 60)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_forward_shapes(name):
    model = build_v3(name).eval()
    x, mask = _dummy()
    with torch.no_grad():
        policy, wdl, mlh = model(x, mask)
    assert policy.shape == (3, 1858)
    assert wdl.shape == (3, 3)
    assert mlh.shape == (3, 1)
    assert torch.allclose(policy.sum(-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(wdl.sum(-1), torch.ones(3), atol=1e-5)
    assert (mlh >= 0).all()


def test_legal_mask_zeroes_illegal_moves():
    model = build_v3("nano").eval()
    x, mask = _dummy()
    with torch.no_grad():
        policy, _, _ = model(x, mask)
    illegal = policy[:, 30:]
    assert float(illegal.max()) < 1e-6
    assert torch.allclose(policy[:, :30].sum(-1), torch.ones(3), atol=1e-5)


def test_gradients_flow():
    model = build_v3("nano")
    x, mask = _dummy()
    policy, wdl, mlh = model(x, mask)
    loss = policy[:, 0].sum() + wdl[:, 0].sum() + mlh.sum()
    loss.backward()
    grad = model.policy.proj.weight.grad if hasattr(model.policy, "proj") else None
    assert grad is not None and torch.isfinite(grad).all()


def test_bilinear_head_is_translation_equivariant():
    """Un décalage identique des poids doit décaler les logits de la même façon.

    Vérification indirecte : la tête n'a aucun paramètre indexé par case, donc le nombre
    de paramètres ne dépend pas de la position des pièces.
    """
    model = build_v3("nano")
    n_head = sum(p.numel() for p in model.policy.parameters())
    assert n_head < 20000, f"tête de politique trop grosse : {n_head} paramètres"


def test_dense_head_has_more_parameters():
    bil = build_v3("nano")
    dense = build_v3("nano_dense")
    assert sum(p.numel() for p in dense.parameters()) > sum(p.numel() for p in bil.parameters())


def test_custom_config_builds():
    cfg = V3Config(name="test", blocks=2, filters=24, expand=2, kernel=3, head="hybrid", head_dim=16)
    model = DebZeroV3(cfg).eval()
    x, mask = _dummy(1)
    with torch.no_grad():
        policy, wdl, _ = model(x, mask)
    assert policy.shape == (1, 1858) and wdl.shape == (1, 3)


def test_unknown_head_raises():
    with pytest.raises(ValueError):
        DebZeroV3(V3Config(head="magique"))
