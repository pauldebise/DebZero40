"""Non-régression : le nouveau chemin d'encodage/inférence doit être identique à l'ancien.

On compare, sur des positions blanches ET noires, les probabilités par coup et le WDL
entre l'implémentation d'origine (copie de baseline-paul) et la nouvelle.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import chess
import numpy as np

NET = "nets/N1_10x64/run_10_64_1_fp32.onnx"

# --- référence : code d'origine figé dans git (baseline-paul)
import subprocess
REF_DIR = ".ref_baseline"
if not os.path.isdir(REF_DIR):
    os.makedirs(REF_DIR, exist_ok=True)
    for f in ["src/utils/net_wrapper.py", "src/utils/board_encoder.py", "src/utils/mapping_out_1858.py"]:
        content = subprocess.run(["git", "show", f"baseline-paul:{f}"], capture_output=True, text=True, check=True).stdout
        dst = os.path.join(REF_DIR, os.path.basename(f))
        with open(dst, "w") as fh:
            fh.write(content)


def load_ref():
    """Charge le NetWrapper d'origine avec ses propres modules isolés."""
    spec_be = importlib.util.spec_from_file_location("ref_board_encoder", os.path.join(REF_DIR, "board_encoder.py"))
    ref_be = importlib.util.module_from_spec(spec_be)
    sys.modules["ref_board_encoder"] = ref_be
    spec_be.loader.exec_module(ref_be)

    spec_map = importlib.util.spec_from_file_location("ref_mapping", os.path.join(REF_DIR, "mapping_out_1858.py"))
    ref_map = importlib.util.module_from_spec(spec_map)
    sys.modules["ref_mapping"] = ref_map
    spec_map.loader.exec_module(ref_map)

    src = open(os.path.join(REF_DIR, "net_wrapper.py")).read()
    src = src.replace("from src.utils.board_encoder import fen_to_planes_int8",
                      "from ref_board_encoder import fen_to_planes_int8")
    src = src.replace("from src.utils.mapping_out_1858 import FlatMapper",
                      "from ref_mapping import FlatMapper")
    path = os.path.join(REF_DIR, "ref_net_wrapper.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec_nw = importlib.util.spec_from_file_location("ref_net_wrapper", path)
    mod = importlib.util.module_from_spec(spec_nw)
    spec_nw.loader.exec_module(mod)
    return mod.NetWrapper


POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    "r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 10",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/4P3/8/8/8/8/k7/4K3 w - - 0 1",
    "8/8/8/8/8/8/k1p5/4K3 b - - 0 1",
]


def main():
    from src.utils.net_wrapper import NetWrapper as NewNW
    RefNW = load_ref()

    # On compare à température neutre des deux côtés. L'ancien code force 8.0 en finale
    # (< 6 pièces) : c'est un choix qu'on supprime volontairement côté recherche, donc on
    # neutralise ce comportement dans la référence pour comparer les encodages.
    ref = RefNW(NET, cpu=True, temperature=1.0)
    ref._get_dynamic_temperature = lambda board: 1.0
    new = NewNW(NET, cpu=True, temperature=1.0)

    worst_p = 0.0
    worst_w = 0.0
    failures = 0

    for fen in POSITIONS:
        b = chess.Board(fen)
        rp, rw, _ = ref.inference(b)
        np_, nw_, _ = new.inference(b)

        rmap = {m: p for p, m in rp}
        nmap = {m: p for p, m in np_}

        if set(rmap) != set(nmap):
            print(f"!! jeux de coups différents pour {fen}")
            print("   manquants:", set(rmap) - set(nmap), "en trop:", set(nmap) - set(rmap))
            failures += 1
            continue

        dp = max(abs(rmap[m] - nmap[m]) for m in rmap)
        dw = float(np.max(np.abs(np.array(rw) - np.array(nw_))))
        worst_p = max(worst_p, dp)
        worst_w = max(worst_w, dw)
        status = "OK " if dp < 1e-6 and dw < 1e-6 else "ECART"
        print(f"{status} {fen[:46]:46s} d_policy={dp:.2e} d_wdl={dw:.2e}")
        if dp >= 1e-6 or dw >= 1e-6:
            failures += 1

    print(f"\nécart max policy={worst_p:.2e} wdl={worst_w:.2e} | positions en écart: {failures}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
