"""Mesure de référence de l'état 'baseline-paul' (avant améliorations).

Usage: python bench/bench_baseline.py [--threads N] [--net PATH] [--seconds S]
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from time import perf_counter


def bench_inference(net_path, n=50):
    from src.utils.net_wrapper import NetWrapper
    net = NetWrapper(net_path, cpu=True)
    b = chess.Board()
    net.inference(b)  # warmup
    t0 = perf_counter()
    for _ in range(n):
        net.inference(b)
    dt = (perf_counter() - t0) / n
    return dt * 1000


def bench_search(net_path, threads, seconds):
    from src.utils.net_wrapper import NetWrapper
    from src.engine.mcts_singlethreaded import MctsGraph as Single
    from src.engine.mcts_multithreaded import MctsGraph as Multi

    net = NetWrapper(net_path, cpu=True)
    board = chess.Board()
    klass = Single if threads == 1 else Multi
    if threads == 1:
        engine = klass(net, syzygy_path=None)
    else:
        engine = klass(net, num_threads=threads, syzygy_path=None)
    engine.search(board)  # warmup

    positions = [
        chess.Board(),  # départ
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        chess.Board("r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 10"),
    ]
    res = []
    for pos in positions:
        nodes_before = 0
        import io, contextlib
        buf = io.StringIO()
        t0 = perf_counter()
        with contextlib.redirect_stdout(buf):
            best, _, pv = engine.search(pos, nodes=None, stop_event=None,
                                        time_manager=_TM(seconds))
        dt = perf_counter() - t0
        out = buf.getvalue().strip().splitlines()
        last = out[-1] if out else ""
        nps = 0
        for tok in last.split():
            pass
        nps_match = [t for t in last.split(" nps ")]
        if len(nps_match) > 1:
            nps = int(nps_match[1].split()[0])
        res.append((pos.fen()[:30], best.uci(), round(dt, 2), nps))
    return res


class _TM:
    def __init__(self, secs):
        self.allocated_time = secs
        self.is_pondering = False
        self.timer_start = perf_counter()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--net", default="nets/N1_10x64/run_10_64_1_fp32.onnx")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--skip-inf", action="store_true")
    a = ap.parse_args()

    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(cwd)
    print("net:", a.net, "threads:", a.threads, "seconds:", a.seconds)
    if not a.skip_inf:
        print(f"inference: {bench_inference(a.net):.2f} ms/pos")
    for fen, move, dt, nps in bench_search(a.net, a.threads, a.seconds):
        print(f"{fen:32s} best={move} time={dt:.2f}s nps={nps}")
