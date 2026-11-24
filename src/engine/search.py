"""
Adaptateur entre le protocole UCI et le moteur MCTS (`mcts_core`).

Conçu pour tourner sur 1 vCPU (VPS). Un booléen d'arrêt est vérifié à chaque simulation,
donc `stop`/`ponderhit` sont pris en compte sans délai.

Points importants :
  * état de répétition : la position reçue par `position ... moves` est conservée avec son
    historique, pour que `is_repetition` du MCTS soit fiable ;
  * un évaluateur met en cache les inférences d'une même position (le réseau est déterministe) ;
  * le temps alloué suit un profil prudent : jamais plus de ~10 % du temps restant.
"""
from __future__ import annotations

import threading
from time import perf_counter
from typing import Dict, List, Optional

import chess
import numpy as np

from src.engine.mcts_core import MCTS, wdl_to_cp
from src.utils.net_wrapper import NetWrapper


class NetEvaluator:
    """Évaluateur réseau sans température, avec cache par clé de position."""

    def __init__(self, net: NetWrapper, max_cache: int = 400_000):
        self.net = net
        self.cache: Dict[int, tuple] = {}
        self.max_cache = max_cache
        self.inferences = 0
        self.inference_time = 0.0

    def __call__(self, board: chess.Board):
        key = chess.polyglot.zobrist_hash(board)
        hit = self.cache.get(key)
        if hit is not None:
            policy, wdl = hit
            return dict(policy), wdl

        t0 = perf_counter()
        proba_moves, raw_wdl, _ = self.net.inference(board)
        self.inference_time += perf_counter() - t0
        self.inferences += 1

        policy = {m: float(p) for p, m in proba_moves}
        wdl = np.array(raw_wdl, dtype=np.float64)

        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[key] = (policy, wdl)
        return dict(policy), wdl

    def clear(self):
        self.cache.clear()


class TimeBudget:
    """Répartition prudente du temps : ~1/25 du temps restant + moitié de l'incrément."""

    MIN_MOVE_TIME = 0.03      # jamais moins de 30 ms de recherche
    SAFETY = 0.85             # marge pour la latence réseau/lichess

    def __init__(self):
        self.allocated_time = 1.0
        self.is_pondering = False
        self.timer_start = perf_counter()

    def allocate(self, params: dict, turn: bool) -> float:
        movetime = params.get("movetime")
        if movetime:
            return max(movetime / 1000.0 * self.SAFETY, 0.001)

        if turn == chess.WHITE:
            our_time = params.get("wtime") or 0
            our_inc = params.get("winc") or 0
        else:
            our_time = params.get("btime") or 0
            our_inc = params.get("binc") or 0

        if our_time <= 0:
            return 1.0

        rest = our_time / 1000.0
        inc = our_inc / 1000.0
        movestogo = params.get("movestogo") or 0

        if movestogo > 0:
            share = rest / max(movestogo, 1)
        else:
            share = rest / 25.0

        target = share + 0.5 * inc
        # Ne jamais dépasser 10 % du temps restant, même en fin de partie.
        target = min(target, 0.10 * rest)
        # Et toujours garder une réserve : au moins 3 coups d'avance en bulles.
        target = min(target, max(rest - 0.05, rest * 0.5))
        return max(min(target, rest * 0.95), self.MIN_MOVE_TIME)


class Search:
    """Pilote une recherche MCTS mono-thread avec arrêt externe (UCI)."""

    def __init__(self, net: Optional[NetWrapper], threads: int = 1, syzygy_path: Optional[str] = None):
        self.evaluator = NetEvaluator(net) if net is not None else None
        self.mcts = MCTS(self.evaluator, dirichlet_epsilon=0.0) if self.evaluator else None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.budget = TimeBudget()
        self.board: Optional[chess.Board] = None
        self.threads = threads
        self.tb_hits = 0
        self.tablebase = None
        self.syzygy_path = syzygy_path
        if syzygy_path:
            try:
                self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
                print(f"info string Syzygy Tablebases found at {syzygy_path}", flush=True)
            except Exception:
                print("info string Warning: Syzygy path not valid", flush=True)

    # ---------------------------------------------------------------- syzygy
    def syzygy_move(self, board: chess.Board) -> Optional[chess.Move]:
        if self.tablebase is None or board.occupied.bit_count() > 5:
            return None
        best_move = None
        best = (-9, 0, 0)
        try:
            for move in board.legal_moves:
                board.push(move)
                try:
                    wdl = self.tablebase.get_wdl(board)
                    dtz = self.tablebase.get_dtz(board)
                finally:
                    board.pop()
                if wdl is None or dtz is None:
                    continue
                score = (-wdl, -abs(dtz) if wdl <= 0 else -abs(dtz))
                if score > best:
                    best = score
                    best_move = move
        except Exception:
            return None
        return best_move

    # ---------------------------------------------------------------- recherche
    def _should_stop(self) -> bool:
        if self.stop_event.is_set():
            return True
        if self.budget.is_pondering:
            return False
        return perf_counter() - self.budget.timer_start >= self.budget.allocated_time

    def _report(self, board: chess.Board):
        m = self.mcts
        nodes = m.root_nodes()
        dt = max(perf_counter() - self.budget.timer_start, 1e-6)
        pv = " ".join(x.uci() for x in m.principal_variation())
        q = m.root_value()
        depth = 0
        if nodes > 1:
            depth = max(int(np.log(nodes) / np.log(max(board.legal_moves.count(), 2))), 1)
        print(
            f"info depth {depth} seldepth {len(m.principal_variation())} score cp {wdl_to_cp(q)} "
            f"nodes {nodes} nps {int(nodes / dt)} tbhits {self.tb_hits} time {int(dt * 1000)} pv {pv}",
            flush=True,
        )

    def _run(self, board: chess.Board, max_nodes: Optional[int]):
        """Recherche avec garde-fou : on doit TOUJOURS répondre un bestmove.

        Un moteur qui ne répond pas à un `go` perd la partie au temps. Position sans coup
        légal, réseau absent, branche imprévue : on répond `bestmove 0000` (nulle UCI)
        plutôt que de laisser le fil mourir en silence.
        """
        try:
            if self.mcts is None:
                self._emit(chess.Move.null(), None)
                return

            # Aucun coup légal (mat ou pat) : `bestmove 0000` est la réponse UCI attendue.
            if board.is_game_over(claim_draw=False) and board.legal_moves.count() == 0:
                self._emit(chess.Move.null(), None)
                return

            # Mate immédiat : inutile de lancer un arbre.
            for move in board.legal_moves:
                board.push(move)
                is_mate = board.is_checkmate()
                board.pop()
                if is_mate:
                    self._emit(move, None)
                    return

            if max_nodes is None:
                def stop():
                    self._should_stop_run += 1
                    if self._should_stop_run % 200 == 0:
                        self._report(board)
                    return self._should_stop()
            else:
                def stop():
                    # `>=` et non `>` : on veut EXACTEMENT max_nodes simulations, et
                    # l'arrêt ne doit pas se déclencher avant la première.
                    if self._should_stop_run >= max_nodes:
                        return True
                    self._should_stop_run += 1
                    return self.stop_event.is_set()

            self._should_stop_run = 0
            self.mcts.run(board, max_nodes=max_nodes if max_nodes else 1_000_000_000, should_stop=stop)

            self._report(board)

            if self.mcts.root is None or not self.mcts.root.children:
                self._emit(chess.Move.null(), None)
                return

            best = self.mcts.best_move()
            pv = self.mcts.principal_variation()
            self._emit(best, pv[1] if len(pv) > 1 else None)

        except Exception as e:  # noqa: BLE001
            # On ne laisse jamais lichess-bot sans réponse.
            print(f"info string search error: {type(e).__name__}: {e}", flush=True)
            fallback = next(iter(board.legal_moves), chess.Move.null())
            self._emit(fallback, None)

    def _emit(self, move: chess.Move, ponder: Optional[chess.Move]):
        out = f"bestmove {move.uci()}"
        if ponder is not None:
            out += f" ponder {ponder.uci()}"
        print(out, flush=True)

    def start(self, board: chess.Board, params: dict):
        self.stop_search()
        self.stop_event.clear()
        self.board = board.copy(stack=True)
        self.budget.is_pondering = bool(params.get("ponder"))
        self.budget.timer_start = perf_counter()
        self.budget.allocated_time = (
            float("inf") if self.budget.is_pondering
            else self.budget.allocate(params, board.turn)
        )

        tb = self.syzygy_move(self.board)
        if tb is not None:
            self.tb_hits += 1
            self._emit(tb, None)
            return

        max_nodes = params.get("nodes")
        self.thread = threading.Thread(target=self._run, args=(self.board, max_nodes), daemon=True)
        self.thread.start()

    def stop_search(self):
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()

    def ponderhit(self):
        if self.budget.is_pondering:
            self.budget.is_pondering = False
            self.budget.timer_start = perf_counter()
            if self.board is not None:
                # Recalcule un budget normal pour la position pondérée.
                self.budget.allocated_time = max(
                    self.budget.allocate({"wtime": 60000, "winc": 0}, self.board.turn), 0.05
                )
