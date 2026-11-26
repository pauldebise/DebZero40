"""Match au temps égal entre deux moteurs, en parallèle.

Différence avec ``bench/match.py`` : ici chaque coup reçoit un budget de **temps**
(``--movetime-ms``) et non un nombre de nœuds. C'est la mesure qui compte pour un bot
en ligne : un réseau plus rapide visite plus de positions dans le même temps.

Chaque ouverture aléatoire est jouée **deux fois**, une avec le candidat en blanc,
une en noir : la variance due à l'ouverture disparaît du résultat.

Exemple :

    python bench/match_time.py \
        --a "python -m src.engine.uci" --a-net nets/N1_6x64/run_6_64_2_fp32.onnx \
        --b "python -m src.engine.uci" --b-net nets/v2/nano_full_fp32.onnx \
        --games 200 --movetime-ms 1000 --parallel 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from multiprocessing import Pool
from typing import List, Optional

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UciEngine:
    """Pilote un moteur UCI en sous-processus, en mode temps ou nœuds."""

    def __init__(
        self,
        name: str,
        argv: List[str],
        cwd: str,
        net: Optional[str] = None,
        movetime_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        env_extra: Optional[dict] = None,
    ):
        self.name = name
        self.movetime_ms = movetime_ms
        self.nodes = nodes
        self.last_nodes = 0
        env = {**os.environ, "PYTHONPATH": cwd, **(env_extra or {})}
        self.proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name Threads value 1")
        if net:
            self._send(f"setoption name NetPath value {net}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, line: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str, limit: int = 400000) -> List[str]:
        assert self.proc.stdout is not None
        out = []
        for _ in range(limit):
            line = self.proc.stdout.readline()
            if not line:
                break
            out.append(line.rstrip())
            if line.strip().startswith(token):
                return out
        return out

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def best_move(self, moves: List[str]) -> str:
        pos = "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        if self.movetime_ms:
            self._send(f"go movetime {self.movetime_ms}")
        else:
            self._send(f"go nodes {self.nodes or 400}")
        lines = self._read_until("bestmove")
        self.last_nodes = 0
        for line in lines:
            if line.startswith("info") and " nodes " in line:
                try:
                    self.last_nodes = int(line.split(" nodes ")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if line.startswith("bestmove"):
                return line.split()[1]
        raise RuntimeError(f"{self.name}: pas de bestmove")

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def random_opening(plies: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    b = chess.Board()
    moves = []
    for _ in range(plies):
        if b.is_game_over():
            break
        m = rng.choice(list(b.legal_moves))
        moves.append(m.uci())
        b.push(m)
    return moves


def _split_env(argv: List[str]):
    env_extra, rest = {}, []
    for tok in argv:
        if not rest and "=" in tok and not tok.startswith("-"):
            k, v = tok.split("=", 1)
            env_extra[k] = v
        else:
            rest.append(tok)
    return env_extra, rest


def play_job(job: dict) -> dict:
    """Joue une partie. ``job`` contient tout le nécessaire (picklable)."""
    opening = job["opening"]
    b_white = job["b_white"]
    movetime = job["movetime"]
    max_plies = job["max_plies"]

    a_env, a_argv = _split_env(job["a_argv"])
    b_env, b_argv = _split_env(job["b_argv"])

    if b_white:
        white_spec = ("B", b_argv, job["b_net"], b_env)
        black_spec = ("A", a_argv, job["a_net"], a_env)
    else:
        white_spec = ("A", a_argv, job["a_net"], a_env)
        black_spec = ("B", b_argv, job["b_net"], b_env)

    w = UciEngine(white_spec[0], white_spec[1], ROOT, white_spec[2], movetime_ms=movetime,
                  env_extra=white_spec[3])
    k = UciEngine(black_spec[0], black_spec[1], ROOT, black_spec[2], movetime_ms=movetime,
                  env_extra=black_spec[3])

    moves = list(opening)
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    w.new_game()
    k.new_game()

    nodes = {"A": 0, "B": 0}
    detail = ""
    outcome = "draw"
    culprit = None

    try:
        while not board.is_game_over(claim_draw=False) and len(moves) < max_plies:
            engine = w if board.turn == chess.WHITE else k
            try:
                uci = engine.best_move(moves)
            except Exception as exc:  # noqa: BLE001
                outcome, culprit, detail = "crash", engine.name, str(exc)[:80]
                break
            nodes[engine.name] += engine.last_nodes
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                outcome, culprit, detail = "illegal", engine.name, uci
                break
            moves.append(uci)
            board.push(mv)
            if board.can_claim_fifty_moves() or board.is_repetition(3) or board.is_insufficient_material():
                outcome = "draw"
                break
        else:
            if board.is_checkmate():
                outcome = "white" if board.turn == chess.BLACK else "black"
            else:
                outcome = "draw"
    finally:
        w.quit()
        k.quit()

    if outcome == "draw":
        pts = 0.5
    elif outcome in ("crash", "illegal"):
        pts = 1.0 if culprit == "A" else 0.0
        detail = f"{outcome} de {culprit}: {detail}"
    elif outcome == "white":
        pts = 1.0 if b_white else 0.0
    else:
        pts = 0.0 if b_white else 1.0

    return {
        "pts": pts,
        "outcome": outcome,
        "detail": detail,
        "plies": len(moves),
        "nodes_a": nodes["A"],
        "nodes_b": nodes["B"],
        "b_white": b_white,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="commande du moteur de référence")
    ap.add_argument("--a-net", default=None)
    ap.add_argument("--b", required=True, help="commande du moteur candidat")
    ap.add_argument("--b-net", default=None)
    ap.add_argument("--games", type=int, default=100, help="nombre total de parties (pair)")
    ap.add_argument("--movetime-ms", type=int, default=1000, help="temps par coup")
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="", help="fichier JSON de résultats")
    args = ap.parse_args()

    n_openings = max(1, args.games // 2)
    openings = [random_opening(args.opening_plies, args.seed + i) for i in range(n_openings)]
    jobs = []
    for idx, opening in enumerate(openings):
        for b_white in (True, False):
            jobs.append(
                {
                    "opening": opening,
                    "b_white": b_white,
                    "movetime": args.movetime_ms,
                    "max_plies": args.max_plies,
                    "a_argv": args.a.split(),
                    "b_argv": args.b.split(),
                    "a_net": args.a_net,
                    "b_net": args.b_net,
                }
            )

    print(
        f"{len(jobs)} parties | {args.movetime_ms} ms/coup | {n_openings} ouvertures "
        f"jouées deux fois | {args.parallel} en parallèle",
        flush=True,
    )

    results = []
    with Pool(processes=args.parallel) as pool:
        for i, res in enumerate(pool.imap_unordered(play_job, jobs), start=1):
            results.append(res)
            if i % 10 == 0 or i == len(jobs):
                partiels = [r["pts"] for r in results]
                print(
                    f"  {i}/{len(jobs)} parties | score candidat "
                    f"{100 * sum(partiels) / len(partiels):.1f} %",
                    flush=True,
                )

    n = len(results)
    score = sum(r["pts"] for r in results) / n
    var = sum((r["pts"] - score) ** 2 for r in results) / max(n - 1, 1)
    se = math.sqrt(var / n) if n > 1 else 0.0
    lo, hi = score - 1.96 * se, score + 1.96 * se

    def elo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -400 * math.log10(1 / p - 1)

    nodes_a = sum(r["nodes_a"] for r in results)
    nodes_b = sum(r["nodes_b"] for r in results)
    plies = sum(r["plies"] for r in results)
    crashes = [r for r in results if r["outcome"] in ("crash", "illegal")]
    wins_b = sum(1 for r in results if r["pts"] == 1.0)
    draws = sum(1 for r in results if r["pts"] == 0.5)
    losses_b = sum(1 for r in results if r["pts"] == 0.0)

    print("\n=== Résultat ===")
    print(f"candidat : {wins_b} victoires, {draws} nulles, {losses_b} défaites sur {n}")
    print(f"score = {score * 100:.1f} %  [IC95 : {lo * 100:.1f} % - {hi * 100:.1f} %]")
    print(f"écart Elo estimé : {elo(score):+.0f}  (IC95 : {elo(lo):+.0f} à {elo(hi):+.0f})")
    print(f"nœuds par coup : référence {nodes_a / max(plies, 1):.0f}, candidat {nodes_b / max(plies, 1):.0f}")
    if crashes:
        print(f"incidents : {len(crashes)} (ex. {crashes[0]['detail'][:70]})")
    if lo > 0.5:
        print("=> le candidat est significativement plus fort")
    elif hi < 0.5:
        print("=> le candidat est significativement plus faible")
    else:
        print("=> pas de différence significative à ce nombre de parties")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {"score": score, "lo": lo, "hi": hi, "elo": elo(score), "results": results},
                fh,
                indent=1,
            )
        print(f"résultats écrits : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
