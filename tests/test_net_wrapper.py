"""Tests du wrapper ONNX sans réseau réel : détection de l'espace de policy et masque."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chess

from src.utils import net_wrapper
from src.utils.net_wrapper import NetWrapper, lc0_index_of
from src.utils.mapping_out_1858 import FlatMapper


class FakeInput:
    def __init__(self, name, shape, type_="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_


class FakeSession:
    """Session ONNX factice : `run` renvoie une policy dérivée du masque si fourni."""

    def __init__(self, inputs, policy=None):
        self._inputs = inputs
        self._policy = policy
        self.feeds = []

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, feed):
        self.feeds.append(feed)
        if self._policy is not None:
            policy = np.asarray(self._policy, dtype=np.float32)
        elif "legal_mask" in feed:
            mask = np.asarray(feed["legal_mask"], dtype=np.float32).reshape(-1)
            total = mask.sum()
            policy = mask / total if total else mask
        else:
            policy = np.zeros(1858, dtype=np.float32)
            policy[0] = 1.0
        wdl = np.array([0.3, 0.4, 0.3], dtype=np.float32)
        return [policy.reshape(1, 1858), wdl.reshape(1, 3)]


def install_fake(monkeypatch, inputs, policy=None):
    session = FakeSession(inputs, policy)

    def factory(path, sess_options=None, providers=None):
        return session

    monkeypatch.setattr(net_wrapper.ort, "InferenceSession", factory)
    return session


def make_wrapper(monkeypatch, inputs, policy=None):
    session = install_fake(monkeypatch, inputs, policy)
    return NetWrapper("fake.onnx"), session


def test_detects_lc0_policy_from_legal_mask(monkeypatch):
    nw, _ = make_wrapper(monkeypatch, [
        FakeInput("input", [1, 8, 8, 17]),
        FakeInput("legal_mask", [1, 1858]),
    ])
    assert nw.input_planes == 17
    assert nw.uses_legal_mask is True
    assert nw.uses_lc0 is True
    print("  legal_mask -> espace LC0")


def test_detects_legacy_policy_without_legal_mask(monkeypatch):
    nw, _ = make_wrapper(monkeypatch, [FakeInput("input", [1, 8, 8, 12])])
    assert nw.input_planes == 12
    assert nw.uses_legal_mask is False
    assert nw.uses_lc0 is False
    print("  pas de legal_mask -> ancienne liste maison")


def test_detection_does_not_depend_on_plane_count(monkeypatch):
    """Un réseau 17 plans sans masque reste sur l'ancienne liste ; l'inverse aussi."""
    nw17, _ = make_wrapper(monkeypatch, [FakeInput("input", [1, 8, 8, 17])])
    assert nw17.uses_lc0 is False

    nw12, _ = make_wrapper(monkeypatch, [
        FakeInput("input", [1, 8, 8, 12]),
        FakeInput("legal_mask", [1, 1858]),
    ])
    assert nw12.uses_lc0 is True
    print("  l'espace de policy ne dépend pas du nombre de plans")


def test_lc0_legal_mask_contains_exactly_legal_indices(monkeypatch):
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    nw, session = make_wrapper(monkeypatch, [
        FakeInput("input", [1, 8, 8, 17]),
        FakeInput("legal_mask", [1, 1858]),
    ])

    proba_moves, wdl, duration = nw.inference(board)
    feed = session.feeds[-1]
    assert "legal_mask" in feed
    mask = np.asarray(feed["legal_mask"]).reshape(-1)
    assert mask.shape == (1858,)

    expected = {lc0_index_of(board, m) for m in board.legal_moves}
    assert set(np.flatnonzero(mask).tolist()) == expected

    # La policy retournée ne contient que des coups légaux, une fois chacun.
    assert {m for _, m in proba_moves} == set(board.legal_moves)
    assert len(proba_moves) == len(expected)
    assert np.isclose(sum(p for p, _ in proba_moves), 1.0)
    assert wdl.shape == (3,) and isinstance(duration, float)
    print(f"  masque LC0 exact ({len(expected)} coups légaux)")


def test_legacy_path_uses_flat_mapper(monkeypatch):
    board = chess.Board()
    policy = np.linspace(0.0, 1.0, 1858, dtype=np.float32)
    nw, session = make_wrapper(monkeypatch, [FakeInput("input", [1, 8, 8, 12])], policy)

    proba_moves, _, _ = nw.inference(board)
    feed = session.feeds[-1]
    assert "legal_mask" not in feed

    fm = FlatMapper()
    expected = {m: float(policy[fm.get_move_index(m.uci())]) for m in board.legal_moves}
    got = {m: p for p, m in proba_moves}
    assert got == expected
    print("  chemin historique : ancienne liste, sans masque")


if __name__ == "__main__":
    print("Ces tests nécessitent pytest (fixture monkeypatch).")
