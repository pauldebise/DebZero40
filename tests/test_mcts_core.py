"""Vérifications du nouveau moteur MCTS (mcts_core) avec un évaluateur factice déterministe."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
from src.engine.mcts_core import MCTS, wdl_to_cp, swap_wdl

CALLS = {"n": 0}


def make_evaluator(mate_in=0):
    """Évaluateur jouet : policy uniforme, valeur 0 (ou mate forcé si mate_in>0)."""
    def evaluate(board):
        CALLS["n"] += 1
        legal = list(board.legal_moves)
        pol = {m: 1.0 / len(legal) for m in legal}
        if mate_in and board.fullmove_number >= mate_in:
            return pol, np.array([0.99, 0.005, 0.005])
        return pol, np.array([0.33, 0.34, 0.33])
    return evaluate


def test_sign_convention():
    """Le Q racine doit rester dans [-1,1] et être symétrique sur une position neutre."""
    eng = MCTS(make_evaluator(), dirichlet_epsilon=0.0)
    b = chess.Board()
    eng.run(b, max_nodes=400, should_stop=lambda: False)
    q = eng.root_value()
    print(f"Q racine (position neutre) = {q:+.4f}")
    assert -0.05 <= q <= 0.05, f"Q attendu ~0, obtenu {q}"


def test_never_illegal():
    eng = MCTS(make_evaluator(), dirichlet_epsilon=0.0)
    b = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    for _ in range(30):
        eng.run(b, max_nodes=300, should_stop=lambda: False)
        mv = eng.best_move()
        assert mv in b.legal_moves, f"coup illégal généré: {mv}"
        b.push(mv)
        if b.is_game_over():
            break
    print("aucun coup illégal sur 30 coups consécutifs (Kiwipete)")


def test_mate_detection():
    """Le MCTS doit trouver le mat en 1 quand la politique le permet."""
    def evaluate(board):
        legal = list(board.legal_moves)
        # donne une forte priorité aux échecs
        raw = []
        for m in legal:
            w = 10.0 if board.gives_check(m) else 1.0
            raw.append((m, w))
        s = sum(w for _, w in raw)
        return {m: w / s for m, w in raw}, np.array([0.33, 0.34, 0.33])

    eng = MCTS(evaluate, dirichlet_epsilon=0.0, mate_heuristic=False)
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    eng.run(b, max_nodes=600, should_stop=lambda: False)
    mv = eng.best_move()
    b.push(mv)
    print(f"mat en 1 cherché -> {mv.uci()} (mat: {b.is_checkmate()})")
    assert b.is_checkmate(), "le MCTS n'a pas trouvé le mat en 1"


def test_repetition_is_draw():
    eng = MCTS(make_evaluator(), dirichlet_epsilon=0.0)
    b = chess.Board()
    for _ in range(6):
        b.push(chess.Move.from_uci("g1f3"))
        b.push(chess.Move.from_uci("g8f6"))
        b.push(chess.Move.from_uci("f3g1"))
        b.push(chess.Move.from_uci("f6g8"))
    print("répétition triple détectée:", b.is_repetition(2))
    assert b.is_repetition(2)
    w = eng.terminal_wdl(b)
    assert w is not None and w[1] == 1.0, f"attendu nulle, obtenu {w}"


def test_swap_wdl_not_mutating():
    a = np.array([0.6, 0.3, 0.1])
    b = swap_wdl(a)
    assert np.allclose(a, [0.6, 0.3, 0.1]), "swap_wdl mute l'entrée !"
    assert np.allclose(b, [0.1, 0.3, 0.6])
    print("swap_wdl: pas de mutation, OK")


def test_cp_conversion():
    print("wdl_to_cp(0) =", wdl_to_cp(0.0))
    assert wdl_to_cp(0.0) == 0
    assert wdl_to_cp(0.5) > 0
    assert wdl_to_cp(-0.5) < 0


def test_stop_event_respected():
    """should_stop() doit être consulté à chaque simulation."""
    eng = MCTS(make_evaluator(), dirichlet_epsilon=0.0)
    n = {"v": 0}
    def stop():
        n["v"] += 1
        return n["v"] > 50
    eng.run(chess.Board(), max_nodes=10_000, should_stop=stop)
    print(f"simulations effectuées avant arrêt: {eng.root_nodes()}")
    assert eng.root_nodes() <= 50


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n--- {t.__name__}")
        t()
    print("\nTOUT OK")
