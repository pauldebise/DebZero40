"""Tests du pipeline LC0 (format V6) -> shards d'entraînement."""

import gzip
import io
import json
import os
import struct
import subprocess
import sys
import tarfile

import chess
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.lc0_moves import LC0_MOVE_TO_INDEX  # noqa: E402
from tools.lc0_to_shards import (  # noqa: E402
    CHUNK_SIZE,
    bit_at,
    board_from_record,
    decode_record,
    ep_candidate_indices,
    make_policy,
    output_planes,
    validate_record,
    wdl_target,
)


PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# Position connue, blanche au trait, avec les quatre droits de roque.
KNOWN_FEN = "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 7 12"


def _set_plane_bit(planes, plane, square):
    """Allume la case ``square`` (0 = a1) du plan LC0 ``plane``."""
    planes[plane] |= 1 << (8 * (square // 8) + (7 - square % 8))


def _move_to_lc0_uci(board, move):
    """Encodage indépendant des conventions LC0 (roque et sous-promotion)."""
    if board.is_castling(move):
        rook_file = 7 if chess.square_file(move.to_square) == 6 else 0
        rook_sq = chess.square(rook_file, chess.square_rank(move.from_square))
        return chess.square_name(move.from_square) + chess.square_name(rook_sq)
    if move.promotion == chess.KNIGHT:
        return chess.square_name(move.from_square) + chess.square_name(move.to_square)
    return move.uci()


def encode_record(board, root_q=0.0, root_d=0.0, visits=1000):
    """Fonction inverse du décodeur : fabrique un enregistrement V6 de 8356 octets.

    Le camp au trait est placé comme « nous » (blancs).  Pour une position noire au trait,
    on passe d'abord par ``board.mirror()``, qui retourne le plateau et échange les couleurs
    — c'est le repère du camp au trait utilisé par LC0.
    """
    canonical = board if board.turn == chess.WHITE else board.mirror()
    assert canonical.turn == chess.WHITE

    planes = [0] * 104
    for square, piece in canonical.piece_map().items():
        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6
        _set_plane_bit(planes, plane, square)

    probabilities = np.full(1858, -1.0, dtype=np.float32)
    legal = [m for m in canonical.legal_moves if not canonical.is_en_passant(m)]
    for move in legal:
        idx = LC0_MOVE_TO_INDEX.get(_move_to_lc0_uci(canonical, move))
        if idx is not None:
            probabilities[idx] = 1.0 / max(len(legal), 1)

    buf = bytearray(CHUNK_SIZE)
    struct.pack_into("<II", buf, 0, 6, 0)
    np.frombuffer(buf, dtype="<f4", count=1858, offset=8)[:] = probabilities
    struct.pack_into("<104Q", buf, 7440, *planes)

    rights = canonical.castling_rights
    castling = (
        bool(rights & chess.BB_A1),
        bool(rights & chess.BB_H1),
        bool(rights & chess.BB_A8),
        bool(rights & chess.BB_H8),
    )
    struct.pack_into("<4B", buf, 8272, *(1 if flag else 0 for flag in castling))
    buf[8276] = 0 if board.turn == chess.WHITE else 1
    buf[8277] = board.halfmove_clock
    buf[8278] = 0

    struct.pack_into(
        "<15f", buf, 8280,
        root_q, 0.0, root_d, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    first_idx = next(iter(sorted(
        LC0_MOVE_TO_INDEX[_move_to_lc0_uci(canonical, m)]
        for m in legal if _move_to_lc0_uci(canonical, m) in LC0_MOVE_TO_INDEX
    )), 0)
    struct.pack_into("<I", buf, 8340, visits)
    struct.pack_into("<HH", buf, 8344, first_idx, first_idx)
    return bytes(buf)


# --------------------------------------------------------------------------------------
# 1. Décodage / reconstruction
# --------------------------------------------------------------------------------------

def test_decode_reconstructs_known_board():
    board = chess.Board(KNOWN_FEN)
    assert board.is_valid()

    record = decode_record(encode_record(board))
    rebuilt = board_from_record(record)

    expected = chess.Board(
        "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 7 1"
    )
    assert rebuilt.fen() == expected.fen()
    assert rebuilt.piece_map() == expected.piece_map()

    # Les quatre droits de roque sont bien reconstruits.
    assert rebuilt.castling_rights & chess.BB_A1
    assert rebuilt.castling_rights & chess.BB_H1
    assert rebuilt.castling_rights & chess.BB_A8
    assert rebuilt.castling_rights & chess.BB_H8

    # Encodage de sortie 17 plans.
    planes = output_planes(record)
    assert planes.shape == (17, 8, 8)
    assert planes.dtype == np.uint8
    assert (planes[12:16] == 1).all()
    assert planes[16, 0, 0] == round(255 * 7 / 100)
    # Plans de pièces : tour blanche a1, roi blanc e1, tour noire a8.
    assert planes[3, 0, 0] == 1
    assert planes[5, 0, 4] == 1
    assert planes[9, 7, 0] == 1


def test_decode_black_to_move_uses_side_to_move_perspective():
    board = chess.Board(KNOWN_FEN)
    board.turn = chess.BLACK
    # Les droits de roque restent sur les mêmes cases ; la position reste valide.
    assert board.is_valid()

    record = decode_record(encode_record(board))
    rebuilt = board_from_record(record)

    # Le décodeur renvoie toujours le camp au trait représenté par les blancs.
    assert rebuilt.turn == chess.WHITE
    expected = board.mirror()
    expected.fullmove_number = 1  # le format ne stocke pas le numéro de coup complet
    assert rebuilt.fen() == expected.fen()


# --------------------------------------------------------------------------------------
# 2. bit_at
# --------------------------------------------------------------------------------------

def test_bit_at_known_cases():
    assert bit_at(1 << 7, 0) == 1       # a1 = bit de poids fort du premier octet
    assert bit_at(1 << 0, 7) == 1       # h1 = bit de poids faible
    assert bit_at(1 << 15, 8) == 1      # a2
    assert bit_at(1 << 6, 1) == 1       # b1
    assert bit_at(1 << 8, 15) == 1      # h2
    assert bit_at(0, 0) == 0
    assert bit_at(1 << 7, 7) == 0

    word = 0
    squares = (0, 7, 8, 19, 63)
    for square in squares:
        word |= 1 << (8 * (square // 8) + (7 - square % 8))
    assert [s for s in range(64) if bit_at(word, s)] == list(squares)


# --------------------------------------------------------------------------------------
# 3. WDL
# --------------------------------------------------------------------------------------

def test_wdl_target_normalisation():
    for root_q, root_d in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0),
                           (0.5, 0.25), (-2.0, -1.0), (2.0, 2.0)]:
        wdl = wdl_target(root_q, root_d)
        assert wdl.shape == (3,)
        assert np.all(wdl >= 0.0) and np.all(wdl <= 1.0)
        assert abs(float(wdl.sum()) - 1.0) < 1e-9

    win, draw, loss = wdl_target(0.0, 1.0)
    assert abs(win) < 1e-12
    assert abs(draw - 1.0) < 1e-12
    assert abs(loss) < 1e-12

    win, draw, loss = wdl_target(0.0, 0.0)
    assert abs(win - 0.5) < 1e-12
    assert abs(draw) < 1e-12
    assert abs(loss - 0.5) < 1e-12


# --------------------------------------------------------------------------------------
# 4. Policy creuse
# --------------------------------------------------------------------------------------

def test_sparse_policy_renormalisation_and_truncation():
    probabilities = np.full(1858, -1.0, dtype=np.float32)
    valid = list(range(100, 170))
    probabilities[valid] = np.linspace(1.0, 0.1, len(valid), dtype=np.float32)

    indices, probs, length = make_policy(probabilities)
    assert length == 64
    assert indices.dtype == np.uint16
    assert probs.dtype == np.float16
    assert len(indices) == len(probs) == length
    assert abs(float(probs.astype(np.float64).sum()) - 1.0) < 2e-3
    assert set(indices.tolist()).issubset(valid)
    # Le coup le plus probable est conservé en tête.
    assert int(indices[0]) == 100
    assert float(probs[0]) >= float(probs[-1])

    # Exclusion d'un index (cas d'une prise en passant).
    indices_ep, _, length_ep = make_policy(probabilities, excluded={100})
    assert length_ep == 64
    assert 100 not in set(indices_ep.tolist())

    # Moins de 64 coups valides : la longueur réelle est respectée.
    small = np.full(1858, -1.0, dtype=np.float32)
    small[[1, 2, 3]] = [0.2, 0.3, 0.5]
    indices, probs, length = make_policy(small)
    assert length == 3
    assert abs(float(probs.astype(np.float64).sum()) - 1.0) < 2e-3
    assert int(indices[0]) == 3

    # Aucun coup valide.
    _, _, length = make_policy(np.full(1858, -1.0, dtype=np.float32))
    assert length == 0


def test_en_passant_index_is_excluded_from_policy_and_validation():
    # Position blanche au trait avec une prise en passant possible (e5xd6 e.p.).
    board = chess.Board(
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    )
    assert board.is_valid()
    ep_move = chess.Move.from_uci("e5d6")
    assert board.is_en_passant(ep_move)

    # L'encodeur ne peut pas représenter la case EP ; on simule la policy LC0 réelle en
    # remettant l'index de la prise en passant à une probabilité positive.
    buf = bytearray(encode_record(board))
    ep_idx = LC0_MOVE_TO_INDEX["e5d6"]
    struct.pack_into("<f", buf, 8 + 4 * ep_idx, 1.0 / 20.0)
    record = decode_record(bytes(buf))

    rebuilt = board_from_record(record)
    assert rebuilt.ep_square is None
    assert ep_idx in ep_candidate_indices(rebuilt)

    _, policy, legal, ep_candidates = validate_record(record)
    assert ep_idx in ep_candidates
    assert ep_idx not in policy
    assert policy == legal  # la validation reste exacte après retrait du coup EP

    indices, _, _ = make_policy(record["probabilities"], ep_candidates)
    assert ep_idx not in set(indices.tolist())


# --------------------------------------------------------------------------------------
# 5. Bout en bout
# --------------------------------------------------------------------------------------

def _write_fake_tar(path, boards):
    payload = b"".join(encode_record(board) for board in boards)
    compressed = gzip.compress(payload)
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo("fake/000000.gz")
        info.size = len(compressed)
        tf.addfile(info, io.BytesIO(compressed))


def test_end_to_end_on_fake_tar(tmp_path):
    boards = [
        chess.Board(),
        chess.Board("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 3 9"),
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ]
    tar_path = tmp_path / "fake.tar"
    out_dir = tmp_path / "shards"
    _write_fake_tar(tar_path, boards)

    script = os.path.join(ROOT, "tools", "lc0_to_shards.py")
    proc = subprocess.run(
        [sys.executable, script, "--tar", str(tar_path), "--out", str(out_dir),
         "--shard-size", "2", "--max-positions", "100", "--min-visits", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Taux d'acceptation" in proc.stdout

    shard_files = sorted(out_dir.glob("shard_*.npz"))
    assert [p.name for p in shard_files] == ["shard_00000.npz", "shard_00001.npz"]

    planes, pol_idx, pol_prob, pol_len, wdl = [], [], [], [], []
    for shard in shard_files:
        with np.load(shard) as data:
            planes.append(data["planes"])
            pol_idx.append(data["pol_idx"])
            pol_prob.append(data["pol_prob"])
            pol_len.append(data["pol_len"])
            wdl.append(data["wdl"])

            assert data["planes"].dtype == np.uint8
            assert data["pol_idx"].dtype == np.uint16
            assert data["pol_prob"].dtype == np.float16
            assert data["pol_len"].dtype == np.uint8
            assert data["wdl"].dtype == np.float16
            n = data["planes"].shape[0]
            assert data["planes"].shape == (n, 17, 8, 8)
            assert data["pol_idx"].shape == (n, 64)
            assert data["pol_prob"].shape == (n, 64)
            assert data["pol_len"].shape == (n,)
            assert data["wdl"].shape == (n, 3)

    planes = np.concatenate(planes)
    pol_idx = np.concatenate(pol_idx)
    pol_prob = np.concatenate(pol_prob)
    pol_len = np.concatenate(pol_len)
    wdl = np.concatenate(wdl)

    assert planes.shape == (3, 17, 8, 8)
    assert np.all(pol_len <= 64)
    for i in range(3):
        length = int(pol_len[i])
        assert abs(float(pol_prob[i, :length].astype(np.float64).sum()) - 1.0) < 2e-3
        assert np.all(pol_idx[i, length:] == 0)
        assert abs(float(wdl[i].astype(np.float64).sum()) - 1.0) < 2e-3

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["positions"] == 3
    assert manifest["params"]["shard_size"] == 2
    assert manifest["params"]["min_visits"] == 0
    assert len(manifest["shards"]) == 2
