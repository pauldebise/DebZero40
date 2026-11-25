"""Vérifications du mapping coup python-chess -> index canonique LC0 (1858)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import chess

from src.utils.net_wrapper import lc0_index_of
from src.utils.lc0_moves import LC0_MOVE_TO_INDEX, LC0_MOVES


def _idx(uci):
    return LC0_MOVE_TO_INDEX[uci]


def test_simple_move():
    board = chess.Board()
    assert lc0_index_of(board, chess.Move.from_uci("e2e4")) == 322
    assert lc0_index_of(board, chess.Move.from_uci("e2e4")) == _idx("e2e4")
    print("  e2e4 -> 322")


def test_castling_uses_king_takes_rook():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    kingside = chess.Move.from_uci("e1g1")
    queenside = chess.Move.from_uci("e1c1")
    assert board.is_castling(kingside) and board.is_castling(queenside)
    assert lc0_index_of(board, kingside) == _idx("e1h1")
    assert lc0_index_of(board, queenside) == _idx("e1a1")

    # Noirs : même encodage canonique après miroir.
    black_board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    assert lc0_index_of(black_board, chess.Move.from_uci("e8g8")) == _idx("e1h1")
    assert lc0_index_of(black_board, chess.Move.from_uci("e8c8")) == _idx("e1a1")
    print("  roques -> e1h1 / e1a1 (blancs et noirs)")


def test_underpromotion_knight_uses_bare_square():
    board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    knight_promo = chess.Move.from_uci("a7a8n")
    queen_promo = chess.Move.from_uci("a7a8q")
    assert knight_promo in board.legal_moves and queen_promo in board.legal_moves

    assert lc0_index_of(board, knight_promo) == _idx("a7a8")
    assert lc0_index_of(board, queen_promo) == _idx("a7a8q")
    assert lc0_index_of(board, knight_promo) != lc0_index_of(board, queen_promo)

    # Noirs : la case nue est miroir-retournée (a2a1 -> a7a8).
    black_board = chess.Board("8/8/8/8/8/8/p7/2K4k b - - 0 1")
    black_knight_promo = chess.Move.from_uci("a2a1n")
    assert black_knight_promo in black_board.legal_moves
    assert lc0_index_of(black_board, black_knight_promo) == _idx("a7a8")
    print("  sous-promotion cavalier -> case nue")


def test_random_positions_indices_unique_and_in_range():
    """Sur 200 positions aléatoires : indices uniques et dans [0, 1858)."""
    rng = random.Random(20261125)
    positions = []
    while len(positions) < 200:
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            continue
        positions.append(board)

    for board in positions:
        legal = list(board.legal_moves)
        seen = set()
        for move in legal:
            idx = lc0_index_of(board, move)
            assert idx is not None, f"coup sans index LC0 : {board.fen()} {move.uci()}"
            assert 0 <= idx < 1858, f"index hors borne : {idx}"
            assert idx not in seen, (
                f"collision d'index {idx} sur {board.fen()} ({move.uci()})"
            )
            seen.add(idx)
        assert len(seen) == len(legal)
    print(f"  {len(positions)} positions, {len(LC0_MOVES)} index LC0, aucune collision")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n--- {t.__name__}")
        t()
    print("\nTOUT OK")
