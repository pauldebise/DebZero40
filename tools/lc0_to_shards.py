#!/usr/bin/env python3
"""Convertit les données d'entraînement Leela Chess Zero (format V6) en shards.

Chaque membre ``*.gz`` d'une archive ``.tar`` contient une suite d'enregistrements
``V6TrainingData`` de 8356 octets (structure compacte ``#pragma pack(1)``).  Le script
reconstruit la position depuis les 13 premiers plans, valide la policy LC0 contre les
coups légaux, puis écrit des fichiers ``.npz`` non compressés prêts pour l'entraînement.

Exemple :

    python tools/lc0_to_shards.py --tar data/train.tar --out data/shards \\
        --shard-size 200000 --max-positions 2000000 --min-visits 0

La position reconstruite est toujours vue du camp au trait : ses pièces sont les plans
0..5 (blancs) et celles de l'adversaire les plans 6..11 (noirs).  C'est exactement le
repère du réseau.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import sys
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Iterator

import chess
import numpy as np

# Le script est lancé comme `python tools/lc0_to_shards.py` : la racine du dépôt n'est
# alors pas forcément dans sys.path, ce qui empêcherait d'importer src.utils.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.lc0_moves import LC0_MOVE_TO_INDEX  # noqa: E402


CHUNK_SIZE = 8356
N_MOVES = 1858
N_PLANES_OUT = 17
MAX_POLICY = 64
PROGRESS_EVERY = 20_000
MIN_ACCEPTANCE = 0.99

# Inversion des bits dans chaque octet (LC0 stocke les plans ainsi).
_BYTE_REVERSE = bytes(int(f"{value:08b}"[::-1], 2) for value in range(256))


def bit_at(word: int, square: int) -> int:
    """Lit la case ``square`` (0 = a1) d'un plan LC0 ``word``."""
    return (word >> (8 * (square // 8) + (7 - (square % 8)))) & 1


def plane_to_bitboard(word: int) -> int:
    """Convertit un mot de plan LC0 en bitboard python-chess (case 0 = a1)."""
    raw = int(word).to_bytes(8, "little")
    return int.from_bytes(raw.translate(_BYTE_REVERSE), "little")


# --------------------------------------------------------------------------------------
# Décodage d'un enregistrement
# --------------------------------------------------------------------------------------

def decode_record(buf: bytes) -> dict:
    """Décode un ``V6TrainingData`` de 8356 octets.

    Renvoie un dictionnaire avec les champs utiles.  ``planes`` contient les 104 mots
    uint64 bruts ; ``planes_bits`` les mêmes plans déjà déballés en ``uint8[104, 8, 8]``
    (``[plan, rangée, colonne]``, rangée 0 = rangée 1).
    """
    if len(buf) != CHUNK_SIZE:
        raise ValueError(f"enregistrement de {len(buf)} octets au lieu de {CHUNK_SIZE}")

    version, input_format = struct.unpack_from("<II", buf, 0)
    probabilities = np.frombuffer(buf, dtype="<f4", count=N_MOVES, offset=8)
    planes = np.frombuffer(buf, dtype="<u8", count=104, offset=7440)

    # Chaque uint64 contient 8 octets (un par rangée), bits inversés dans chaque octet.
    # np.unpackbits(bitorder="big") redonne directement l'ordre a1..h1 dans l'octet.
    plane_bytes = planes.view(np.uint8).reshape(104, 8)
    planes_bits = np.unpackbits(plane_bytes, axis=1).reshape(104, 8, 8)

    off = 8272
    castling = struct.unpack_from("<4B", buf, off)  # us_ooo, us_oo, them_ooo, them_oo
    side_to_move = buf[off + 4]
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
        "version": version,
        "input_format": input_format,
        "probabilities": probabilities,
        "planes": planes,
        "planes_bits": planes_bits,
        "castling": castling,
        "side_to_move": side_to_move,
        "rule50": rule50,
        "invariance": invariance,
        "root_q": root_q,
        "best_q": best_q,
        "root_d": root_d,
        "best_d": best_d,
        "root_m": root_m,
        "best_m": best_m,
        "plies_left": plies_left,
        "result_q": result_q,
        "result_d": result_d,
        "played_q": played_q,
        "played_d": played_d,
        "played_m": played_m,
        "orig_q": orig_q,
        "orig_d": orig_d,
        "orig_m": orig_m,
        "visits": visits,
        "played_idx": played_idx,
        "best_idx": best_idx,
    }


def _planes_to_bitboards(planes_bits: np.ndarray) -> list[int]:
    """Convertit les 12 plans de pièces en bitboards python-chess."""
    packed = np.packbits(planes_bits[:12].reshape(12, 64), axis=1, bitorder="little")
    return [int.from_bytes(packed[p].tobytes(), "little") for p in range(12)]


def board_from_record(record: dict) -> chess.Board:
    """Reconstruit la board du point de vue du camp au trait (représenté par les blancs)."""
    bbs = _planes_to_bitboards(record["planes_bits"])

    board = chess.Board(None)
    board.pawns = bbs[0] | bbs[6]
    board.knights = bbs[1] | bbs[7]
    board.bishops = bbs[2] | bbs[8]
    board.rooks = bbs[3] | bbs[9]
    board.queens = bbs[4] | bbs[10]
    board.kings = bbs[5] | bbs[11]

    board.occupied_co[chess.WHITE] = bbs[0] | bbs[1] | bbs[2] | bbs[3] | bbs[4] | bbs[5]
    board.occupied_co[chess.BLACK] = bbs[6] | bbs[7] | bbs[8] | bbs[9] | bbs[10] | bbs[11]
    board.occupied = board.occupied_co[chess.WHITE] | board.occupied_co[chess.BLACK]
    board.promoted = 0

    board.turn = chess.WHITE
    board.ep_square = None
    board.halfmove_clock = int(record["rule50"])
    board.fullmove_number = 1

    rights = 0
    castling = record["castling"]
    if castling[0]:
        rights |= chess.BB_A1
    if castling[1]:
        rights |= chess.BB_H1
    if castling[2]:
        rights |= chess.BB_A8
    if castling[3]:
        rights |= chess.BB_H8
    board.castling_rights = rights
    return board


# --------------------------------------------------------------------------------------
# Coups au format LC0
# --------------------------------------------------------------------------------------

def move_to_lc0_uci(move: chess.Move, board: chess.Board) -> str:
    """Convertit un coup python-chess en chaîne LC0 (roque = roi prend sa tour)."""
    if board.is_castling(move):
        rook_file = 7 if chess.square_file(move.to_square) == 6 else 0
        rook_sq = chess.square(rook_file, chess.square_rank(move.from_square))
        return chess.square_name(move.from_square) + chess.square_name(rook_sq)
    if move.promotion == chess.KNIGHT:
        # Pas de slot dédié : la sous-promotion en cavalier tombe sur le coup sans suffixe.
        return chess.square_name(move.from_square) + chess.square_name(move.to_square)
    return move.uci()


def legal_move_indices(board: chess.Board) -> set[int]:
    """Index LC0 des coups légaux (les prises en passant sont écartées)."""
    out: set[int] = set()
    for move in board.legal_moves:
        if board.is_en_passant(move):
            continue
        idx = LC0_MOVE_TO_INDEX.get(move_to_lc0_uci(move, board))
        if idx is not None:
            out.add(idx)
    return out


def ep_candidate_indices(board: chess.Board) -> set[int]:
    """Index des coups qui pourraient être des prises en passant.

    La case EP n'est pas stockée dans ce format : la board reconstruite n'a donc jamais
    de droit EP.  Un pion du camp au trait (blancs) sur la 5e rangée, avec un pion adverse
    adjacent et une case d'arrivée vide, ne peut être qu'une prise en passant.  On retire
    son index de la policy : il n'est pas reconstructible depuis les plans.
    """
    out: set[int] = set()
    for from_sq in board.pieces(chess.PAWN, chess.WHITE):
        if chess.square_rank(from_sq) != 4:
            continue
        from_file = chess.square_file(from_sq)
        for delta in (-1, 1):
            to_file = from_file + delta
            if not 0 <= to_file < 8:
                continue
            to_sq = chess.square(to_file, 5)
            if board.piece_at(to_sq) is not None:
                continue
            neighbour = board.piece_at(chess.square(to_file, 4))
            if (neighbour is not None and neighbour.piece_type == chess.PAWN
                    and neighbour.color == chess.BLACK):
                idx = LC0_MOVE_TO_INDEX.get(
                    chess.square_name(from_sq) + chess.square_name(to_sq)
                )
                if idx is not None:
                    out.add(idx)
    return out


def validate_record(record: dict) -> tuple[chess.Board, set[int], set[int], set[int]]:
    """Reconstruit la board et compare la policy LC0 aux coups légaux.

    Renvoie ``(board, policy_indices, legal_indices, ep_candidates)``.
    """
    board = board_from_record(record)
    legal = legal_move_indices(board)
    ep_candidates = ep_candidate_indices(board)
    policy = {int(i) for i in np.flatnonzero(record["probabilities"] >= 0)}
    policy -= ep_candidates
    return board, policy, legal, ep_candidates


# --------------------------------------------------------------------------------------
# Cibles d'entraînement
# --------------------------------------------------------------------------------------

def output_planes(record: dict) -> np.ndarray:
    """Encode les 17 plans de sortie, ``uint8[17, 8, 8]``."""
    out = np.zeros((N_PLANES_OUT, 8, 8), dtype=np.uint8)
    out[:12] = record["planes_bits"][:12]

    castling = record["castling"]
    if castling[0]:
        out[12] = 1  # grand roque disponible (nous)
    if castling[1]:
        out[13] = 1  # petit roque disponible (nous)
    if castling[2]:
        out[14] = 1  # grand roque disponible (adversaire)
    if castling[3]:
        out[15] = 1  # petit roque disponible (adversaire)

    rule50 = min(int(record["rule50"]), 100)
    out[16] = int(round(255 * rule50 / 100))
    return out


def make_policy(probabilities: np.ndarray,
                excluded: Iterable[int] = ()) -> tuple[np.ndarray, np.ndarray, int]:
    """Construit la cible policy creuse (64 coups max, renormalisée à 1)."""
    mask = probabilities >= 0
    excluded_set = set(int(i) for i in excluded)
    if excluded_set:
        for idx in excluded_set:
            if 0 <= idx < probabilities.shape[0]:
                mask[idx] = False

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return (np.zeros(0, dtype=np.uint16),
                np.zeros(0, dtype=np.float16),
                0)

    values = probabilities[indices].astype(np.float64)
    order = np.argsort(-values, kind="stable")[:MAX_POLICY]
    indices = indices[order]
    values = values[order]

    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        values = np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    else:
        values = values / total

    return (indices.astype(np.uint16),
            values.astype(np.float16),
            int(indices.size))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def wdl_target(root_q: float, root_d: float) -> np.ndarray:
    """Cible WDL normalisée ``(win, draw, loss)`` depuis la recherche LC0."""
    draw = _clamp01(root_d)
    win = _clamp01((1.0 + float(root_q)) / 2.0 - draw / 2.0)
    loss = _clamp01(1.0 - win - draw)
    total = win + draw + loss
    if not np.isfinite(total) or total <= 0.0:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    return np.array([win / total, draw / total, loss / total], dtype=np.float64)


# --------------------------------------------------------------------------------------
# Lecture en flux du tar
# --------------------------------------------------------------------------------------

def _warn(message: str) -> None:
    print(f"AVERTISSEMENT: {message}", file=sys.stderr)


def _iter_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> Iterator[tuple[int, bytes]]:
    """Itère les enregistrements d'un membre gzip, en flux.

    Une erreur de décompression est signalée puis le membre est abandonné ; les positions
    déjà lues restent valides.
    """
    gz = None
    try:
        fileobj = tf.extractfile(member)
        if fileobj is None:
            _warn(f"{member.name}: membre illisible, ignoré")
            return
        gz = gzip.GzipFile(fileobj=fileobj, mode="rb")
    except Exception as exc:  # noqa: BLE001
        _warn(f"{member.name}: impossible d'ouvrir le membre ({exc}), ignoré")
        return

    carry = b""
    position_index = 0
    try:
        while True:
            block = gz.read(CHUNK_SIZE * 1024)
            if not block:
                break
            data = carry + block if carry else block
            complete = len(data) // CHUNK_SIZE
            offset = 0
            for _ in range(complete):
                yield position_index, data[offset:offset + CHUNK_SIZE]
                position_index += 1
                offset += CHUNK_SIZE
            carry = data[offset:]
        if carry:
            _warn(f"{member.name}: {len(carry)} octets résiduels ignorés (membre tronqué)")
    except Exception as exc:  # noqa: BLE001
        _warn(f"{member.name}: membre corrompu après {position_index} positions ({exc})")
    finally:
        try:
            gz.close()
        except Exception:  # noqa: BLE001
            pass


def iter_lc0_records(tar_path: str) -> Iterator[tuple[int, bytes]]:
    """Itère ``(demi-coup, enregistrement)`` depuis un tar LC0, sans le charger en mémoire."""
    try:
        fileobj = open(tar_path, "rb")
    except OSError as exc:
        _warn(f"impossible d'ouvrir {tar_path}: {exc}")
        return

    tf = None
    try:
        try:
            tf = tarfile.open(fileobj=fileobj, mode="r|")
        except (tarfile.TarError, OSError) as exc:
            _warn(f"{tar_path}: tar illisible ({exc})")
            return
        try:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue
                yield from _iter_member(tf, member)
        except Exception as exc:  # noqa: BLE001
            _warn(f"{tar_path}: erreur de lecture du tar ({exc})")
    finally:
        if tf is not None:
            try:
                tf.close()
            except Exception:  # noqa: BLE001
                pass
        fileobj.close()


# --------------------------------------------------------------------------------------
# Écriture des shards
# --------------------------------------------------------------------------------------

class ShardWriter:
    """Accumule les positions et écrit ``shard_00000.npz``, ``shard_00001.npz``, ..."""

    def __init__(self, out_dir: str, shard_size: int):
        if shard_size <= 0:
            raise ValueError("shard_size doit être strictement positif")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.shard_index = 0
        self.total = 0
        self.shards: list[tuple[str, int]] = []
        self._init_buffer()

    def _init_buffer(self) -> None:
        capacity = min(self.shard_size, 1024)
        self._capacity = max(1, capacity)
        self._count = 0
        self._planes = np.empty((self._capacity, N_PLANES_OUT, 8, 8), dtype=np.uint8)
        self._pol_idx = np.empty((self._capacity, MAX_POLICY), dtype=np.uint16)
        self._pol_prob = np.empty((self._capacity, MAX_POLICY), dtype=np.float16)
        self._pol_len = np.empty((self._capacity,), dtype=np.uint8)
        self._wdl = np.empty((self._capacity, 3), dtype=np.float16)

    def _grow(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_capacity = min(self.shard_size, max(needed, self._capacity * 2))
        if new_capacity < needed:
            new_capacity = needed

        planes = np.empty((new_capacity, N_PLANES_OUT, 8, 8), dtype=np.uint8)
        planes[:self._count] = self._planes[:self._count]
        pol_idx = np.empty((new_capacity, MAX_POLICY), dtype=np.uint16)
        pol_idx[:self._count] = self._pol_idx[:self._count]
        pol_prob = np.empty((new_capacity, MAX_POLICY), dtype=np.float16)
        pol_prob[:self._count] = self._pol_prob[:self._count]
        pol_len = np.empty((new_capacity,), dtype=np.uint8)
        pol_len[:self._count] = self._pol_len[:self._count]
        wdl = np.empty((new_capacity, 3), dtype=np.float16)
        wdl[:self._count] = self._wdl[:self._count]

        self._capacity = new_capacity
        self._planes = planes
        self._pol_idx = pol_idx
        self._pol_prob = pol_prob
        self._pol_len = pol_len
        self._wdl = wdl

    def add(self, planes: np.ndarray, pol_idx: np.ndarray, pol_prob: np.ndarray,
            pol_len: int, wdl: np.ndarray) -> None:
        self._grow(self._count + 1)
        i = self._count

        self._planes[i] = planes
        self._pol_idx[i].fill(0)
        self._pol_prob[i].fill(0)
        self._pol_idx[i, :pol_len] = pol_idx
        self._pol_prob[i, :pol_len] = pol_prob
        self._pol_len[i] = pol_len
        self._wdl[i] = wdl

        self._count += 1
        if self._count >= self.shard_size:
            self.flush()

    def flush(self) -> str | None:
        if self._count == 0:
            return None
        name = f"shard_{self.shard_index:05d}.npz"
        path = os.path.join(self.out_dir, name)
        n = self._count
        np.savez(
            path,
            planes=self._planes[:n],
            pol_idx=self._pol_idx[:n],
            pol_prob=self._pol_prob[:n],
            pol_len=self._pol_len[:n],
            wdl=self._wdl[:n],
        )
        self.shard_index += 1
        self.total += n
        self.shards.append((name, n))
        self._init_buffer()
        return name

    def close(self) -> None:
        self.flush()


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------

@dataclass
class Stats:
    read: int = 0
    accepted: int = 0
    rejected: int = 0
    filtered: int = 0
    bad_version: int = 0
    decode_errors: int = 0
    examples: list[tuple[str, int, int, int, int]] = field(default_factory=list)


def convert(args: argparse.Namespace) -> tuple[Stats, ShardWriter, float]:
    """Exécute la conversion et renvoie les statistiques, le writer et le taux d'acceptation."""
    stats = Stats()
    writer = ShardWriter(args.out, args.shard_size)
    max_positions = args.max_positions if args.max_positions and args.max_positions > 0 else None

    start = time.monotonic()
    records = iter_lc0_records(args.tar)
    try:
        for ply, buf in records:
            stats.read += 1
            if stats.read % PROGRESS_EVERY == 0:
                elapsed = max(time.monotonic() - start, 1e-9)
                print(
                    f"[{stats.read:,} positions lues] {stats.read / elapsed:,.0f} pos/s | "
                    f"acceptées {stats.accepted:,} | rejetées {stats.rejected:,}",
                    flush=True,
                )

            try:
                record = decode_record(buf)
            except Exception as exc:  # noqa: BLE001
                stats.rejected += 1
                stats.decode_errors += 1
                if stats.decode_errors <= 3:
                    _warn(f"enregistrement indécodable ({exc})")
                continue

            if record["version"] != 6:
                stats.rejected += 1
                stats.bad_version += 1
                if stats.bad_version <= 3:
                    _warn(f"version {record['version']} ignorée (attendu : 6)")
                continue

            if record["visits"] < args.min_visits:
                stats.filtered += 1
                continue
            if ply < args.min_ply:
                stats.filtered += 1
                continue

            try:
                board, policy, legal, ep_candidates = validate_record(record)
            except Exception as exc:  # noqa: BLE001
                stats.rejected += 1
                if len(stats.examples) < 5:
                    stats.examples.append((f"erreur: {exc}", 0, 0, 0, 0))
                continue

            if policy != legal:
                stats.rejected += 1
                if len(stats.examples) < 5:
                    stats.examples.append((
                        board.fen(),
                        len(policy),
                        len(legal),
                        len(legal - policy),
                        len(policy - legal),
                    ))
                continue

            stats.accepted += 1
            pol_idx, pol_prob, pol_len = make_policy(record["probabilities"], ep_candidates)
            writer.add(
                output_planes(record),
                pol_idx,
                pol_prob,
                pol_len,
                wdl_target(record["root_q"], record["root_d"]),
            )

            if max_positions is not None and stats.accepted >= max_positions:
                break
    finally:
        records.close()
        writer.close()

    validated = stats.accepted + stats.rejected
    acceptance = stats.accepted / validated if validated else 0.0
    return stats, writer, acceptance


def write_manifest(args: argparse.Namespace, stats: Stats, writer: ShardWriter,
                   acceptance: float) -> str:
    manifest = {
        "positions": writer.total,
        "shards": [{"file": name, "positions": count} for name, count in writer.shards],
        "params": {
            "tar": os.path.abspath(args.tar),
            "out": os.path.abspath(args.out),
            "shard_size": args.shard_size,
            "max_positions": args.max_positions,
            "min_visits": args.min_visits,
            "min_ply": args.min_ply,
        },
        "date": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "read": stats.read,
            "accepted": stats.accepted,
            "rejected": stats.rejected,
            "filtered": stats.filtered,
            "bad_version": stats.bad_version,
            "decode_errors": stats.decode_errors,
            "acceptance_rate": acceptance,
        },
    }
    path = os.path.join(args.out, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def _print_summary(stats: Stats, writer: ShardWriter, acceptance: float) -> None:
    validated = stats.accepted + stats.rejected
    print()
    print(f"Positions lues      : {stats.read:,}")
    print(f"  filtrées          : {stats.filtered:,} (min-visits / min-ply)")
    print(f"  rejetées          : {stats.rejected:,}")
    print(f"  acceptées         : {stats.accepted:,}")
    if validated:
        print(f"Taux d'acceptation  : {acceptance * 100:.2f} % ({stats.accepted:,}/{validated:,})")
    else:
        print("Taux d'acceptation  : n/a (aucune position validée)")
    print(f"Shards écrits       : {len(writer.shards)} ({writer.total:,} positions)")
    for example in stats.examples:
        print(f"  écart : {example}")

    if validated == 0:
        _warn("aucune position validée : vérifier le fichier d'entrée.")
    elif acceptance < MIN_ACCEPTANCE:
        _warn(
            f"taux d'acceptation {acceptance * 100:.2f} % < {MIN_ACCEPTANCE * 100:.0f} % : "
            "les données ou le décodage posent problème."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convertit des fichiers d'entraînement Leela Chess Zero (format V6) "
            "en shards .npz prêts pour l'entraînement."
        )
    )
    parser.add_argument("--tar", required=True, help="archive .tar contenant des membres .gz")
    parser.add_argument("--out", required=True, help="répertoire de sortie des shards")
    parser.add_argument("--shard-size", type=int, default=200000,
                        help="positions par shard (défaut : 200000)")
    parser.add_argument("--max-positions", type=int, default=2000000,
                        help="nombre maximum de positions acceptées, 0 = illimité "
                             "(défaut : 2000000)")
    parser.add_argument("--min-visits", type=int, default=0,
                        help="ignorer les positions sous ce nombre de visites (défaut : 0)")
    parser.add_argument("--min-ply", type=int, default=0,
                        help="ignorer les positions avant ce demi-coup du membre gzip "
                             "(défaut : 0)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.shard_size <= 0:
        parser.error("--shard-size doit être strictement positif")

    stats, writer, acceptance = convert(args)
    write_manifest(args, stats, writer, acceptance)
    _print_summary(stats, writer, acceptance)

    success = bool(writer.shards) and acceptance >= MIN_ACCEPTANCE
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
