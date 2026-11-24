"""Sonde 2 : gestion du temps, sécurité de board en multithread, comportement en fin de partie."""
import sys, os, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
from time import perf_counter

print("=== 1. time_scheduler : temps alloué vs temps restant ===")
from src.engine.uci import time_scheduler
cases = [
    ("départ 10min+2s", dict(side_to_move=chess.WHITE, wtime=600000, btime=600000, winc=2000, binc=2000)),
    ("10s+0 (bulles)", dict(side_to_move=chess.WHITE, wtime=10000, btime=10000, winc=0, binc=0)),
    ("3s+0 restant", dict(side_to_move=chess.WHITE, wtime=3000, btime=3000, winc=0, binc=0)),
    ("1s+0 restant", dict(side_to_move=chess.WHITE, wtime=1000, btime=1000, winc=0, binc=0)),
    ("0.8s+0 restant", dict(side_to_move=chess.WHITE, wtime=800, btime=800, winc=0, binc=0)),
    ("movestogo=3, 10s", dict(side_to_move=chess.WHITE, wtime=10000, btime=10000, winc=0, binc=0, movestogo=3)),
]
for name, kw in cases:
    t = time_scheduler(**kw)
    rest = kw.get('wtime')
    print(f"  {name:22s} -> alloué {t:6.2f}s (reste {rest/1000:5.2f}s)  {'!! SURTEMPS' if t > rest/1000 else ''}")

print("\n=== 2. Multithread : partage du board muté entre workers ===")
import inspect
from src.engine import mcts_multithreaded as mt
src = inspect.getsource(mt.MctsGraph.search)
print("  search crée une board par worker ?", "board.copy()" in inspect.getsource(mt.MctsGraph._worker_thread))
print("  root_board passé aux workers :", "args=(board," in src)
print("  verrou autour de la descente :", "with self.lock:" in inspect.getsource(mt.MctsGraph._worker_thread))

print("\n=== 3. Comportement sur des positions de mat en 1 (threads=1) ===")
from src.utils.net_wrapper import NetWrapper
from src.engine.mcts_singlethreaded import MctsGraph as Single
net = NetWrapper("nets/N1_10x64/run_10_64_1_fp32.onnx", cpu=True)
eng = Single(net, syzygy_path=None)

class TM:
    def __init__(s, secs):
        s.allocated_time = secs; s.is_pondering = False; s.timer_start = perf_counter()

mats = {
    "mat en 1 (Qh5#)": ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", ["e1e8"]),
    "mat en 1 cloué": ("r1b2k1r/ppp1bppp/8/1B1Q4/8/8/PPP2PPP/2KR4 w - - 0 1", ["d5d8"]),
    "dame pendante (Nd5 attaque Qd8? non)": ("r2qkb1r/pp2pppp/2n2n2/3p4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6", []),
}
import io, contextlib
for name, (fen, expected) in mats.items():
    b = chess.Board(fen)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        best, _, pv = eng.search(b, time_manager=TM(1.5))
    flag = ""
    if expected:
        flag = "OK" if best.uci() in expected else f"!! ATTENDU {expected}"
    print(f"  {name:38s} -> {best.uci()} {flag}")

print("\n=== 4. Coup 'nul' d'un réseau qui ignore les sous-promotions / roques ===")
b = chess.Board("4k3/8/8/8/8/8/6r1/R3K2R w KQ - 0 1")
print("  roque blanc O-O légal :", chess.Move.from_uci("e1g1") in b.legal_moves)
