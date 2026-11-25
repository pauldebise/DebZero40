"""Tests de l'encodage étendu (17 plans) et de sa compatibilité avec l'ancien (12 plans)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chess
from src.utils.board_encoder import (
    board_to_planes_int8, PIECE_PLANES, EXTRA_PLANES, TOTAL_PLANES,
)


def _old_encoder(board):
    """Réplique exacte de l'ancien encodage 12 plans (référence de compatibilité)."""
    rank_of = [chess.square_rank(sq) for sq in range(64)]
    file_of = [chess.square_file(sq) for sq in range(64)]
    planes = np.zeros((8, 8, 12), dtype=np.int8)
    black_turn = board.turn == chess.BLACK
    for square, piece in board.piece_map().items():
        rank = rank_of[square]
        if black_turn:
            rank = 7 - rank
        channel = piece.piece_type - 1 + (0 if piece.color == board.turn else 6)
        planes[rank, file_of[square], channel] = 1
    return planes


def test_shapes():
    b = chess.Board()
    assert board_to_planes_int8(b).shape == (8, 8, PIECE_PLANES)
    assert board_to_planes_int8(b, extended=True).shape == (8, 8, TOTAL_PLANES)
    assert TOTAL_PLANES == PIECE_PLANES + EXTRA_PLANES == 17
    # L'ancien chemin garde le dtype historique ; l'étendu doit contenir 0..255.
    assert board_to_planes_int8(b).dtype == np.int8
    assert board_to_planes_int8(b, extended=True).dtype == np.uint8
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


def test_rule50_plane_increases_with_halfmove_clock():
    """Le plan 16 vaut round(255 * min(rule50, 100) / 100) sur TOUTES les cases."""
    clocks = [0, 1, 7, 25, 50, 51, 99, 100, 150]
    previous = -1
    for hc in clocks:
        b = chess.Board(f"8/8/8/8/8/8/8/K6k w - - {hc} 1")
        p = board_to_planes_int8(b, extended=True)
        plane = p[:, :, 16]
        expected = int(round(255 * min(hc, 100) / 100))
        assert (plane == expected).all(), (hc, plane[0, 0], expected)
        assert expected >= previous, "le plan 16 doit être croissant avec halfmove_clock"
        previous = expected
    print("  plan 16 (compteur 50 coups) croissant et exact")


def test_en_passant_no_longer_encoded():
    """Plus de plan EP : à plateau/roques/compteur identiques, la case EP ne change rien."""
    with_ep = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    without_ep = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert with_ep.ep_square is not None and without_ep.ep_square is None
    assert np.array_equal(
        board_to_planes_int8(with_ep, extended=True),
        board_to_planes_int8(without_ep, extended=True),
    ), "la case EP ne doit plus être encodée"
    print("  plan de prise en passant supprimé")


def test_backward_compatible_pieces_planes():
    """extended=False doit produire exactement les mêmes 12 plans qu'avant."""
    import random
    random.seed(3)
    checked = 0
    for _ in range(20):
        b = chess.Board()
        for _ in range(random.randint(0, 30)):
            if b.is_game_over():
                break
            b.push(random.choice(list(b.legal_moves)))
        short = board_to_planes_int8(b)
        long_ = board_to_planes_int8(b, extended=True)
        assert short.dtype == np.int8
        assert np.array_equal(short, long_[:, :, :12]), "les 12 plans de pièces ont changé"
        assert np.array_equal(short, _old_encoder(b)), "l'ancien encodage 12 plans a changé"
        checked += 1
    assert checked > 0
    print(f"  plans de pièces inchangés ({checked} positions aléatoires)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n--- {t.__name__}")
        t()
    print("\nTOUT OK")
