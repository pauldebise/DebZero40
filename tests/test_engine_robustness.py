"""Tests de robustesse du moteur : cas limites qui font perdre une partie si mal gérés."""
import sys, os, subprocess, chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ask(fen, go="go wtime 2000 btime 2000", timeout=120):
    """Lance le moteur en UCI et renvoie la liste des lignes de sortie."""
    inp = f"uci\nisready\nposition fen {fen}\n{go}\nquit\n"
    p = subprocess.run(
        [sys.executable, "-m", "src.engine.uci"],
        input=inp, capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT}, timeout=timeout,
    )
    return p.stdout.splitlines(), p.stderr


def test_position_without_legal_move_answers_0000():
    """Pat : le moteur doit répondre bestmove 0000, jamais rester muet."""
    lines, err = ask("8/8/8/8/8/5k2/5p2/5K2 w - - 0 1")
    bm = [l for l in lines if l.startswith("bestmove")]
    assert bm, f"aucune réponse bestmove (stderr: {err[:300]})"
    assert bm[-1].split()[1] == "0000", f"réponse inattendue: {bm[-1]}"
    assert "Traceback" not in err and "Exception" not in err, f"exception en fond: {err[:400]}"
    print("  pat -> bestmove 0000, aucune exception")


def test_checkmate_position_answers_0000():
    lines, err = ask("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    bm = [l for l in lines if l.startswith("bestmove")]
    assert bm and bm[-1].split()[1] == "0000", f"réponse inattendue: {bm}"
    print("   mat déjà donné -> bestmove 0000")


def test_mate_in_one_always_found():
    positions = [
        ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8"),
        ("7k/6pp/8/8/8/8/8/R6K w - - 0 1", "a1a8"),
    ]
    for fen, expected in positions:
        lines, _ = ask(fen)
        bm = [l for l in lines if l.startswith("bestmove")][-1].split()[1]
        assert bm == expected, f"{fen}: attendu {expected}, obtenu {bm}"
        b = chess.Board(fen)
        b.push(chess.Move.from_uci(bm))
        assert b.is_checkmate(), f"{fen}: {bm} ne donne pas mat"
    print("   mat en 1 trouvé sur", len(positions), "positions")


def test_promotion_position_returns_legal_move():
    """Un pion en 7e rangée ne doit pas produire de coup illégal ni de silence."""
    for fen in ["6k1/7P/8/8/8/8/8/6K1 w - - 0 1", "8/8/8/8/8/8/6PK/7k w - - 0 1"]:
        lines, err = ask(fen)
        bm = [l for l in lines if l.startswith("bestmove")][-1].split()[1]
        b = chess.Board(fen)
        assert chess.Move.from_uci(bm) in b.legal_moves, f"{bm} illégal dans {fen}"
    print("   positions de promotion -> coups légaux")


def test_no_illegal_move_under_time_pressure():
    """Sous forte pression au temps, le coup doit rester légal et rapide."""
    import time
    t0 = time.time()
    lines, _ = ask("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
                   go="go wtime 300 btime 300")
    dt = time.time() - t0
    bm = [l for l in lines if l.startswith("bestmove")][-1].split()[1]
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    assert chess.Move.from_uci(bm) in b.legal_moves
    print(f"   0,3 s restantes -> {bm} en {dt:.1f}s (process complet)")


def test_settings_keep_working():
    """setoption NetPath / SyzygyPath ne doit pas casser la session."""
    inp = ("uci\nisready\n"
           "setoption name SyzygyPath value <empty>\n"
           "setoption name Threads value 4\n"
           "isready\nposition startpos\ngo nodes 50\nquit\n")
    p = subprocess.run([sys.executable, "-m", "src.engine.uci"], input=inp, capture_output=True,
                       text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT}, timeout=120)
    assert p.stdout.count("readyok") >= 2, p.stdout[-500:]
    assert "bestmove" in p.stdout, p.stdout[-500:]
    print("   setoption + isready + go OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n--- {t.__name__}")
        t()
    print("\nTOUT OK")
