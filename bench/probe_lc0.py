"""Sonde : décoder de vraies données d'entraînement LC0 (format v6) et le valider.

Méthode de validation sans faire confiance au format : dans un chunk LC0, la policy vaut
-1 sur tous les coups illégaux et une valeur >= 0 sur les coups légaux. On reconstruit la
position depuis les plans, on calcule l'ensemble des index des coups légaux, et on vérifie
qu'il coïncide exactement avec l'ensemble {i : prob[i] >= 0}. Si ça coïncide sur des
milliers de positions, le décodage est prouvé correct.
"""
import argparse
import io
import struct
import tarfile
import time

import chess
import numpy as np

from src.utils.lc0_moves import LC0_MOVES, LC0_MOVE_TO_INDEX

CHUNK_SIZE = 8356
PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def parse_chunk(buf: bytes):
    """Décode un V6TrainingData (packed, 8356 octets)."""
    version, input_format = struct.unpack_from("<II", buf, 0)
    probs = np.frombuffer(buf, dtype="<f4", count=1858, offset=8)
    planes = np.frombuffer(buf, dtype="<u8", count=104, offset=7440)
    off = 8272
    castling = struct.unpack_from("<4B", buf, off)  # us_ooo, us_oo, them_ooo, them_oo
    stm_or_ep = buf[off + 4]
    rule50 = buf[off + 5]
    invariance = buf[off + 6]
    off = 8280
    (root_q, best_q, root_d, best_d, root_m, best_m, plies_left,
     result_q, result_d, played_q, played_d, played_m,
     orig_q, orig_d, orig_m) = struct.unpack_from("<15f", buf, off)
    off += 60
    (visits,) = struct.unpack_from("<I", buf, off)
    played_idx, best_idx = struct.unpack_from("<HH", buf, off + 4)

    return {
        "version": version, "input_format": input_format, "probabilities": probs,
        "planes": planes, "castling": castling, "stm_or_ep": stm_or_ep, "rule50": rule50,
        "invariance": invariance, "root_q": root_q, "root_d": root_d, "best_q": best_q,
        "best_d": best_d, "result_q": result_q, "result_d": result_d, "visits": visits,
        "played_idx": played_idx, "best_idx": best_idx,
    }


def plane_to_bitboard(word: int) -> int:
    """LC0 écrit les plans avec les bits inversés dans chaque octet."""
    out = 0
    for byte_index in range(8):
        b = (word >> (8 * byte_index)) & 0xFF
        b = int(f"{b:08b}"[::-1], 2)
        out |= b << (8 * byte_index)
    return out


def board_from_planes(planes, castling):
    """Reconstruit une board où le camp au trait est représenté par les blancs."""
    board = chess.Board(None)
    board.clear()
    for plane in range(6):
        bb = plane_to_bitboard(int(planes[plane]))
        for sq in chess.scan_reversed(bb):
            board.set_piece_at(sq, chess.Piece(PIECE_ORDER[plane], chess.WHITE))
    for plane in range(6, 12):
        bb = plane_to_bitboard(int(planes[plane]))
        for sq in chess.scan_reversed(bb):
            board.set_piece_at(sq, chess.Piece(PIECE_ORDER[plane - 6], chess.BLACK))
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    if castling[0]:
        board.castling_rights |= chess.BB_A1
    if castling[1]:
        board.castling_rights |= chess.BB_H1
    if castling[2]:
        board.castling_rights |= chess.BB_A8
    if castling[3]:
        board.castling_rights |= chess.BB_H8
    return board


def legal_index_set(board: chess.Board):
    """Index LC0 des coups légaux, avec les deux conventions de LC0 :

    - le roque est encodé « roi prend sa propre tour » (e1h1 / e1a1), pas e1g1 / e1c1 ;
    - la sous-promotion en cavalier n'a pas de slot dédié : elle tombe sur la case sans
      suffixe (a7a8), parce que l'indice de pièce déborde du masque de 2 bits.
    """
    out = set()
    for move in board.legal_moves:
        if board.is_castling(move):
            # `move.to_square` vaut g1/c1 en UCI normal ; LC0 encode le roque comme le roi
            # prenant sa propre tour, donc h1/a1 (même rangée que le roi).
            rook_file = 7 if move.to_square > move.from_square else 0
            rook_sq = chess.square(rook_file, chess.square_rank(move.from_square))
            uci = chess.square_name(move.from_square) + chess.square_name(rook_sq)
        elif move.promotion == chess.KNIGHT:
            uci = chess.square_name(move.from_square) + chess.square_name(move.to_square)
        else:
            uci = move.uci()
        idx = LC0_MOVE_TO_INDEX.get(uci)
        if idx is not None:
            out.add(idx)
    return out


def ep_from_chunk(chunk):
    """Case de prise en passant.

    En format canonique, le champ `side_to_move_or_enpassant` contient un masque de colonne
    (un seul bit). Dans notre repère (le camp au trait est représenté par les blancs), la case
    visée est toujours sur la 3e rangée.
    """
    mask = chunk["stm_or_ep"]
    if mask == 0:
        return None
    file_index = (mask & -mask).bit_length() - 1
    if file_index > 7:
        return None
    return chess.square(file_index, 5)


def analyse(chunk):
    planes = chunk["planes"]
    board = board_from_planes(planes, chunk["castling"])
    ep = None  # ce format stocke le trait, pas la case EP
    if ep is not None:
        board.ep_square = ep
    prob_set = {int(i) for i in np.nonzero(chunk["probabilities"] >= 0)[0]}
    legal = legal_index_set(board)
    return board, prob_set, legal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--max-games", type=int, default=50)
    args = ap.parse_args()

    stats = {"games": 0, "positions": 0, "match": 0, "mismatch": 0,
             "extra_legal": 0, "missing_legal": 0, "bad_version": 0, "no_probs": 0}
    examples = []
    t0 = time.time()

    with open(args.tar, "rb") as fh:
        tf = tarfile.open(fileobj=fh, mode="r|")
        for member in tf:
            if stats["games"] >= args.max_games:
                break
            if not member.isfile():
                continue
            raw = tf.extractfile(member).read()
            # Les jeux sont stockés en gzip dans le tar.
            if raw[:2] == b"\x1f\x8b":
                import gzip
                raw = gzip.decompress(raw)
            data = raw
            n = len(data) // CHUNK_SIZE
            stats["games"] += 1
            for i in range(n):
                buf = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
                chunk = parse_chunk(buf)
                stats["positions"] += 1
                if chunk["version"] != 6:
                    stats["bad_version"] += 1
                    continue
                if not (chunk["probabilities"] >= 0).any():
                    stats["no_probs"] += 1
                    continue
                board, prob_set, legal = analyse(chunk)
                if prob_set == legal:
                    stats["match"] += 1
                else:
                    stats["mismatch"] += 1
                    stats["extra_legal"] += len(legal - prob_set)
                    stats["missing_legal"] += len(prob_set - legal)
                    if len(examples) < 3:
                        examples.append((board.fen(), len(prob_set), len(legal),
                                         len(legal - prob_set), len(prob_set - legal)))

    dt = time.time() - t0
    total = stats["match"] + stats["mismatch"]
    print(f"jeux lus : {stats['games']} | positions : {stats['positions']} | {dt:.1f}s")
    print(f"versions != 6 : {stats['bad_version']} | sans policy : {stats['no_probs']}")
    if total:
        print(f"COÏNCIDENCE EXACTE : {stats['match']}/{total} = {100*stats['match']/total:.2f} %")
        print(f"coups légaux manquants dans la policy : {stats['missing_legal']}")
        print(f"index en trop dans la policy : {stats['extra_legal']}")
    for ex in examples:
        print("  écart :", ex)


if __name__ == "__main__":
    main()
