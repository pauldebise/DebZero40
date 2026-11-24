"""Tests de l'encodage étendu (17 plans) et de sa compatibilité avec l'ancien (12 plans)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chess
from src.utils.board_encoder import (
    board_to_planes_int8, PIECE_PLANES, EXTRA_PLANES, TOTAL_PLANES,
)


def test_shapes():
    b = chess.Board()
    assert board_to_planes_int8(b).shape == (8, 8, PIECE_PLANES)
    assert board_to_planes_int8(b, extended=True).shape == (8, 8, TOTAL_PLANES)
    assert TOTAL_PLANES == PIECE_PLANES + EXTRA_PLANES == 17
    print("  formes 12 et 17 plans OK")


def test_castling_rights_are_encoded():
    """Le défaut d'origine : plateau identique + droits différents => encodage identique.
    Ici on vérifie que l'encodage étendu les DISTINGUE."""
    a = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 8 3")
    c = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1")

    pa, pb, pc = (board_to_planes_int8(x, extended=True) for x in (a, b, c))
    assert not np.array_equal(pa, pb), "droits absents non distingués"
    assert not np.array_equal(pa, pc), "grand roque absent non distingué"
    # les 12 plans de pièces restent identiques
    assert np.array_equal(pa[:, :, :12], pb[:, :, :12])
    print("  droits de roque distincts (plans 12-15)")


def test_castling_channels_semantics():
    """Canaux 12/13 = nos roques, 14/15 = ceux de l'adversaire."""
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    p = board_to_planes_int8(b, extended=True)
    for ch in (12, 13, 14, 15):
        assert p[:, :, ch].sum() == 64, f"canal {ch} doit être à 1 partout"

    b2 = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1")
    p2 = board_to_planes_int8(b2, extended=True)
    assert p2[:, :, 12].sum() == 0, "grand roque blanc perdu doit être à 0"
    assert p2[:, :, 13].sum() == 64
    assert p2[:, :, 14].sum() == 64 and p2[:, :, 15].sum() == 64
    print("  sémantique des canaux ok")


def test_black_perspective():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    p = board_to_planes_int8(b, extended=True)
    # Aux noirs de jouer : canaux 12/13 = roques NOIRS, 14/15 = roques blancs.
    # FEN « KQk » retire le petit roque noir (k), pas le grand : le canal 13 doit tomber.
    b_no_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQq - 0 1")
    p2 = board_to_planes_int8(b_no_rights, extended=True)
    assert p[:, :, 12].sum() == 64, "grand roque noir présent"
    assert p2[:, :, 12].sum() == 64, "grand roque noir doit rester"
    assert p[:, :, 13].sum() == 64 and p2[:, :, 13].sum() == 0, "petit roque noir doit changer"
    assert p2[:, :, 14].sum() == 64 and p2[:, :, 15].sum() == 64, "roques blancs (adverses) intacts"
    print("  perspective noire ok")


def test_en_passant_plane():
    b = chess.Board()
    for m in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        b.push(chess.Move.from_uci(m))
    p = board_to_planes_int8(b, extended=True)
    assert b.ep_square is not None
    assert p[:, :, 16].sum() == 1, "exactement une case EP"
    rank, file = np.argwhere(p[:, :, 16] == 1)[0]
    assert (rank, file) == (chess.square_rank(b.ep_square), chess.square_file(b.ep_square))
    print("  case EP encodée au bon endroit")


def test_backward_compatible_pieces_planes():
    """extended=False doit produire exactement les mêmes 12 plans qu'avant."""
    import random
    random.seed(3)
    for _ in range(20):
        b = chess.Board()
        for _ in range(random.randint(0, 30)):
            if b.is_game_over():
                break
            b.push(random.choice(list(b.legal_moves)))
        short = board_to_planes_int8(b)
        long_ = board_to_planes_int8(b, extended=True)
        assert np.array_equal(short, long_[:, :, :12]), "les 12 plans de pièces ont changé"
    print("  plans de pièces inchangés (20 positions aléatoires)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n--- {t.__name__}")
        t()
    print("\nTOUT OK")
