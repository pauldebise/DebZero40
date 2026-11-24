"""Match A/B entre deux moteurs UCI, en pur Python (pas de cutechess nécessaire).

Usage :
    python bench/match.py --a "baseline" --b "new" --games 20 --nodes 400

Chaque moteur est lancé comme un sous-processus UCI. Les coups sont tirés au hasard au
début (petit livre d'ouvertures aléatoire) pour éviter de rejouer 20 fois la même partie.

Sortie : score de B contre A (V=1, N=0.5, D=0) avec intervalle de confiance à 95 %.
"""
import argparse
import math
import os
import random
import subprocess
import sys
from typing import List, Optional

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UciEngine:
    def __init__(self, name: str, argv: List[str], cwd: str, nodes: int, net: Optional[str] = None,
                 env_extra: Optional[dict] = None):
        self.name = name
        self.nodes = nodes
        env = {**os.environ, "PYTHONPATH": cwd, **(env_extra or {})}
        self.proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        if net:
            self._send(f"setoption name NetPath value {net}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, line: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str, timeout_lines: int = 200000) -> List[str]:
        assert self.proc.stdout is not None
        out = []
        for _ in range(timeout_lines):
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
        self._send(f"go nodes {self.nodes}")
        lines = self._read_until("bestmove")
        for line in lines:
            if line.startswith("bestmove"):
                return line.split()[1]
        raise RuntimeError(f"{self.name}: pas de bestmove (sortie: {lines[-3:]})")

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def random_opening(plies: int) -> List[str]:
    b = chess.Board()
    moves = []
    for _ in range(plies):
        if b.is_game_over():
            break
        m = random.choice(list(b.legal_moves))
        moves.append(m.uci())
        b.push(m)
    return moves


def play_game(white: UciEngine, black: UciEngine, opening_plies: int, max_plies: int = 300):
    moves = random_opening(opening_plies)
    b = chess.Board()
    for uci in moves:
        b.push(chess.Move.from_uci(uci))

    white.new_game()
    black.new_game()

    while not b.is_game_over(claim_draw=False) and len(moves) < max_plies:
        engine = white if b.turn == chess.WHITE else black
        try:
            uci = engine.best_move(moves)
        except Exception as e:  # noqa: BLE001
            return ("loss_by_crash", engine.name, str(e))
        mv = chess.Move.from_uci(uci)
        if mv not in b.legal_moves:
            return ("loss_by_illegal", engine.name, uci)
        moves.append(uci)
        b.push(mv)
        # nulle automatique : 50 coups, répétition triple, matériel insuffisant
        if b.can_claim_fifty_moves() or b.is_repetition(3) or b.is_insufficient_material():
            return ("draw", None, "règle de nulle")

    res = b.result(claim_draw=True)
    if res == "1-0":
        return ("white", None, "")
    if res == "0-1":
        return ("black", None, "")
    return ("draw", None, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="commande du moteur A (référence)")
    ap.add_argument("--b", required=True, help="commande du moteur B (candidat)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    a_argv = args.a.split()
    b_argv = args.b.split()

    # Supporte un préfixe d'environnement dans la commande : "FOO=bar python ..."
    def split_env(argv):
        env_extra, rest = {}, []
        for tok in argv:
            if not rest and "=" in tok and not tok.startswith("-"):
                k, v = tok.split("=", 1)
                env_extra[k] = v
            else:
                rest.append(tok)
        return env_extra, rest

    a_env, a_argv = split_env(a_argv)
    b_env, b_argv = split_env(b_argv)

    score_b = 0.0
    results = []

    for i in range(args.games):
        # alternance des couleurs
        if i % 2 == 0:
            w = UciEngine("B", b_argv, ROOT, args.nodes, env_extra=b_env)
            k = UciEngine("A", a_argv, ROOT, args.nodes, env_extra=a_env)
        else:
            w = UciEngine("A", a_argv, ROOT, args.nodes, env_extra=a_env)
            k = UciEngine("B", b_argv, ROOT, args.nodes, env_extra=b_env)

        try:
            outcome, culprit, detail = play_game(w, k, args.opening_plies)
        finally:
            w.quit()
            k.quit()

        b_is_white = w.name == "B"
        if outcome == "draw":
            pts = 0.5
        elif outcome in ("loss_by_illegal", "loss_by_crash"):
            pts = 1.0 if culprit == "A" else 0.0
            detail = f"{outcome} de {culprit}: {detail[:60]}"
        elif outcome == "white":
            pts = 1.0 if b_is_white else 0.0
        else:
            pts = 0.0 if b_is_white else 1.0

        score_b += pts
        results.append(pts)
        print(f"  partie {i+1:2d}/{args.games} (B en {'blanc' if b_is_white else 'noir':5s}) "
              f"-> {outcome:16s} pts_B={pts:.1f}  {detail}", flush=True)

    n = len(results)
    mean = score_b / n
    # intervalle de confiance à 95 % (approximation normale sur le score)
    var = sum((r - mean) ** 2 for r in results) / max(n - 1, 1)
    se = math.sqrt(var / n) if n > 1 else 0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se

    print(f"\n=== Résultat ===")
    print(f"B ({args.b}) marque {score_b:.1f}/{n} contre A ({args.a})")
    print(f"score B = {mean*100:.1f} % [IC95 : {lo*100:.1f} % - {hi*100:.1f} %]")
    if lo > 0.5:
        print("=> B est significativement meilleur que A")
    elif hi < 0.5:
        print("=> B est significativement PLUS FAIBLE que A")
    else:
        print("=> pas de différence significative à ce nombre de parties")


if __name__ == "__main__":
    main()
