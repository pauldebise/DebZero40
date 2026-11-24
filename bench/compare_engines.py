"""Comparaison de valeur (Q racine) entre baseline et nouveau moteur, même réseau, même nb de nœuds.

Sert à vérifier qu'il n'y a pas d'inversion de signe : les deux doivent donner
une évaluation comparable sur les mêmes positions.
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
from time import perf_counter

NET = "nets/N1_10x64/run_10_64_1_fp32.onnx"

POSITIONS = {
    "départ": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "italienne": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "blancs gagnants": "r1bq1rk1/ppp2ppp/2np4/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQR1K1 w - - 0 1",
    "noirs gagnants": "rnb1kbnr/pppp1ppp/8/4p3/3qP3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
    "mat en 1": "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
}


def new_engine(wdl_flip=False):
    from src.utils.net_wrapper import NetWrapper
    from src.engine.mcts_core import MCTS, wdl_to_cp

    net = NetWrapper(NET, cpu=True, temperature=1.0)

    def evaluate(board):
        pol, wdl, _ = net.inference(board)
        arr = np.array(wdl, dtype=np.float64)
        if wdl_flip:
            arr = np.array([arr[2], arr[1], arr[0]])
        return {m: p for p, m in pol}, arr

    return MCTS(evaluate, dirichlet_epsilon=0.0), wdl_to_cp


def baseline_engine():
    from src.utils.net_wrapper import NetWrapper
    from src.engine.mcts_singlethreaded import MctsGraph

    net = NetWrapper(NET, cpu=True, temperature=1.0)
    return MctsGraph(net, syzygy_path=None)


class TM:
    def __init__(self, seconds):
        self.allocated_time = seconds
        self.is_pondering = False
        self.timer_start = perf_counter()


def run_baseline(eng, board, nodes):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        best, _, pv = eng.search(board, nodes=nodes, time_manager=TM(600))
    # reconstruire la valeur depuis la ligne info finale
    last = [l for l in buf.getvalue().splitlines() if l.startswith("info depth")]
    cp = None
    if last:
        toks = last[-1].split()
        if "score" in toks:
            cp = int(toks[toks.index("score") + 2])
    return best, cp


def run_new(eng, board, nodes):
    eng.run(board, max_nodes=nodes, should_stop=lambda: False)
    return eng.best_move(), eng.root_value()


if __name__ == "__main__":
    NODES = int(os.environ.get("NODES", "600"))
    base = baseline_engine()
    new, new_cp = new_engine()

    print(f"nœuds demandés: {NODES}\n")
    print(f"{'position':18s} {'baseline':>22s} {'nouveau':>24s}")
    for name, fen in POSITIONS.items():
        b = chess.Board(fen)
        bmv, bcp = run_baseline(base, b, NODES)
        nmv, nq = run_new(new, b, NODES)
        print(f"{name:18s} {str(bmv.uci())+' cp='+str(bcp):>22s} "
              f"{str(nmv.uci())+' q='+format(nq,'+.3f')+' cp='+str(new_cp(nq)):>24s}")
        if bmv != nmv:
            print(f"   -> coups différents (légal nouveau: {nmv in b.legal_moves})")
