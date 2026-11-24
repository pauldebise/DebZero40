"""Test de bout en bout du protocole UCI : on pilote le moteur comme le ferait lichess-bot."""
import sys, os, subprocess, time, select

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run_session(commands, expect_bestmove=True, timeout=60):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    proc = subprocess.Popen(
        [PY, "-m", "src.engine.uci"], cwd=ROOT, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    out_lines = []
    t0 = time.time()
    for cmd in commands:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

        if cmd.startswith("go"):
            # attendre bestmove
            while time.time() - t0 < timeout:
                line = proc.stdout.readline()
                if not line:
                    break
                out_lines.append(line.rstrip())
                if line.startswith("bestmove"):
                    break
        elif cmd in ("uci", "isready"):
            target = "uciok" if cmd == "uci" else "readyok"
            while time.time() - t0 < timeout:
                line = proc.stdout.readline()
                if not line:
                    break
                out_lines.append(line.rstrip())
                if line.strip() == target:
                    break
        else:
            time.sleep(0.05)

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    try:
        rest, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        rest = ""
    if rest:
        out_lines.extend(rest.splitlines())

    return out_lines


def bestmoves(lines):
    return [l.split()[1] for l in lines if l.startswith("bestmove")]


def main():
    print("=== 1. handshake uci/isready ===")
    lines = run_session(["uci", "isready"])
    assert "uciok" in lines, lines
    assert "readyok" in lines, lines
    assert any("id name DebZero40" in l for l in lines)
    threads = [l for l in lines if "option name Threads" in l]
    print("  ", threads[0] if threads else "option Threads absente")
    assert threads and "default 1" in threads[0], "Threads doit par défaut à 1 (1 vCPU en pratique)"
    print("  OK")

    print("\n=== 2. position startpos + go wtime/btime ===")
    lines = run_session(["uci", "isready", "position startpos", "go wtime 60000 btime 60000 winc 1000 binc 1000"])
    bm = bestmoves(lines)
    print("   bestmove:", bm)
    assert bm, "aucun bestmove renvoyé"
    import chess
    assert chess.Move.from_uci(bm[0]) in chess.Board().legal_moves
    info = [l for l in lines if l.startswith("info depth")]
    assert info, "aucune ligne info"
    print("   dernière info:", info[-1][:110])
    print("  OK")

    print("\n=== 3. match en cours (position avec coups) ===")
    lines = run_session([
        "uci", "isready",
        "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5",
        "go wtime 5000 btime 5000",
    ])
    bm = bestmoves(lines)
    print("   bestmove:", bm)
    import chess
    b = chess.Board()
    for m in "e2e4 e7e5 g1f3 b8c6 f1b5".split():
        b.push(chess.Move.from_uci(m))
    assert chess.Move.from_uci(bm[0]) in b.legal_moves, f"{bm[0]} illégal"
    print("  OK")

    print("\n=== 4. go nodes N (limite déterministe) ===")
    lines = run_session(["uci", "isready", "position startpos", "go nodes 200"])
    bm = bestmoves(lines)
    info = [l for l in lines if l.startswith("info depth")][-1]
    print("   ", info[:100])
    assert bm
    print("  OK")

    print("\n=== 5. stop doit interrompre proprement ===")
    import threading
    env = dict(os.environ); env["PYTHONPATH"] = ROOT
    proc = subprocess.Popen([PY, "-m", "src.engine.uci"], cwd=ROOT, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    proc.stdin.write("position startpos\n"); proc.stdin.flush()
    proc.stdin.write("go infinite\n"); proc.stdin.flush()
    time.sleep(1.5)
    t0 = time.time()
    proc.stdin.write("stop\n"); proc.stdin.flush()
    got = None
    while time.time() - t0 < 5:
        line = proc.stdout.readline()
        if not line:
            break
        if line.startswith("bestmove"):
            got = line.strip()
            break
    dt = time.time() - t0
    print(f"   bestmove après stop en {dt:.2f}s : {got}")
    assert got is not None, "stop n'a pas produit de bestmove"
    proc.stdin.write("quit\n"); proc.stdin.flush()
    proc.communicate(timeout=5)
    print("  OK")

    print("\n=== 6. mat en 1 détecté immédiatement ===")
    lines = run_session(["uci", "isready", "position fen 6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "go wtime 1000 btime 1000"])
    bm = bestmoves(lines)
    print("   bestmove:", bm)
    import chess
    assert bm[0] == "e1e8", f"mat en 1 non trouvé: {bm}"
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    b.push(chess.Move.from_uci(bm[0]))
    assert b.is_checkmate()
    print("  OK")

    print("\nTOUS LES TESTS UCI PASSENT")


if __name__ == "__main__":
    main()
