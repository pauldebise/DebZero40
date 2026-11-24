"""Sonde : vérifie les bugs suspectés dans le moteur baseline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
import numpy as np
from src.utils.mapping_out_1858 import FlatMapper

fm = FlatMapper()

print("=== 1. Mapper : couverture des coups légaux sur 500 positions aléatoires ===")
import random
random.seed(0)
missing = {}
total = 0
for _ in range(500):
    b = chess.Board()
    for _ in range(random.randint(0, 40)):
        if b.is_game_over():
            break
        mv = random.choice(list(b.legal_moves))
        b.push(mv)
    for m in b.legal_moves:
        total += 1
        if fm.get_move_index(m.uci()) is None:
            missing[m.uci()] = missing.get(m.uci(), 0) + 1
print(f"coups légaux testés: {total}, introuvables dans le mapper: {sum(missing.values())}")
print("exemples:", list(missing.items())[:10])

print("\n=== 2. Collision promoteur : a7a8q vs a7b8q vs autres ===")
for u in ["a7a8", "a7a8q", "a7a8n", "a7a8r", "a7a8b"]:
    print(u, "->", fm.get_move_index(u))

print("\n=== 3. Mapping d'un coup noir : board.mirror() ===")
b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
mb = b.mirror()
print("noir joue e7e5, miroir =", chess.Move.from_uci("e2e4"), "index:", fm.get_move_index("e2e4"))
print("legal dans miroir ?", chess.Move.from_uci("e2e4") in mb.legal_moves)

print("\n=== 4. Policy de réseau : somme sur les coups légaux (départ) ===")
from src.utils.net_wrapper import NetWrapper
net = NetWrapper("nets/N1_10x64/run_10_64_1_fp32.onnx", cpu=True)
probs, wdl, dt = net.inference(chess.Board())
s = sum(p for p, m in probs)
print(f"somme policy sur 20 coups légaux = {s:.4f}, wdl = {[round(float(x),3) for x in wdl]}")

print("\n=== 5. Température appliquée sur la policy (temp=1.5 par défaut) ===")
net1 = NetWrapper("nets/N1_10x64/run_10_64_1_fp32.onnx", cpu=True, temperature=1.0)
p1, _, _ = net1.inference(chess.Board())
top = sorted(probs, reverse=True)[:5]
top1 = sorted(p1, reverse=True)[:5]
print("temp 1.5 top5:", [round(float(p), 4) for p, m in top])
print("temp 1.0 top5:", [round(float(p), 4) for p, m in top1])

print("\n=== 6. Heuristique finale : score à partir du WDL ===")
for w in [(0.1, 0.3, 0.6), (0.5, 0.4, 0.1), (0.99, 0.005, 0.005)]:
    arr = np.array(w)
    val = arr[0] - arr[2]
    cp = int(-np.log((1.0 - val + 1e-5) / (1.0 + val + 1e-5)) * 300)
    print(f"wdl={w} -> val={val:.2f} cp={cp} (négatif = l'adversaire gagne ?)")
