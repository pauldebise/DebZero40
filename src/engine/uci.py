"""
Point d'entrée UCI du moteur DebZero40.

Le protocole est géré ici ; la recherche vit dans `src/engine/search.py`
(qui s'appuie sur `src/engine/mcts_core.py`). Tout reste en Python.

Compatibilité : options `Threads`, `NetPath`, `SyzygyPath`, `Ponder` et le
comportement `position` / `go` / `stop` / `ponderhit` restent ceux attendus par
lichess-bot et les interfaces UCI standard.
"""
import os
import sys

import chess

from src.engine.search import Search
from src.utils.net_wrapper import NetWrapper


def get_absolute_path(relative_path):
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


DEFAULT_NET = "run_6_64_2_fp32.onnx"


def parse_go_params(command_list):
    """Analyse une commande `go ...` en dictionnaire de paramètres."""
    params = {
        "wtime": None, "btime": None, "winc": 0, "binc": 0,
        "movestogo": 0, "movetime": None, "ponder": False,
        "nodes": None, "searchmoves": None,
    }

    it = iter(command_list[1:])
    for arg in it:
        if arg == "ponder":
            params["ponder"] = True
        elif arg == "moves":
            break
        elif arg == "searchmoves":
            moves = []
            for move_uci in it:
                try:
                    moves.append(chess.Move.from_uci(move_uci))
                except ValueError:
                    break
            params["searchmoves"] = moves
        elif arg in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes"):
            try:
                params[arg] = int(next(it))
            except (StopIteration, ValueError):
                break

    return params


def position(command_str: str) -> chess.Board:
    """Construit la board en conservant l'historique (nécessaire pour les répétitions)."""
    args = command_str.split()

    if args and args[0] == "position":
        args = args[1:]

    if not args:
        return chess.Board()

    if args[0] == "fen":
        if "moves" in args:
            moves_idx = args.index("moves")
            fen = " ".join(args[1:moves_idx])
        else:
            fen = " ".join(args[1:])
        try:
            board = chess.Board(fen)
        except ValueError:
            board = chess.Board()
    else:
        board = chess.Board()

    if "moves" in args:
        moves_idx = args.index("moves")
        for move_uci in args[moves_idx + 1:]:
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                continue
            if move in board.legal_moves:
                board.push(move)

    return board


def uci_header():
    print("id name DebZero40", flush=True)
    print("id author Paul Debise", flush=True)
    print("option name Ponder type check default false")
    print("option name Threads type spin default 1 min 1 max 64")
    print(f"option name NetPath type string default {DEFAULT_NET}")
    print("option name SyzygyPath type string default <empty>")
    print("uciok", flush=True)


def load_net(netpath: str) -> NetWrapper:
    return NetWrapper(get_absolute_path(netpath), cpu=True)


def main():
    netpath = DEFAULT_NET
    try:
        net = load_net(netpath)
    except Exception as e:  # noqa: BLE001
        print("--------------------------------------------------")
        print("Fatal error encountered during start :")
        print(e)
        print("--------------------------------------------------")
        print(f"The engine searches for the nets here : {get_absolute_path(netpath)}")
        print("Please check if nets are correctly placed.")
        print("--------------------------------------------------")
        sys.exit(1)

    current_syzygy_path = None
    search = Search(net, threads=1, syzygy_path=current_syzygy_path)
    state_board = chess.Board()

    while True:
        try:
            command = input()
        except (EOFError, KeyboardInterrupt):
            search.stop_search()
            break

        if not command.strip():
            continue

        cmd = command.split(" ")[0]

        if cmd == "uci":
            uci_header()

        elif cmd == "isready":
            print("readyok", flush=True)

        elif cmd == "setoption":
            parts = command.split()
            try:
                name = parts[parts.index("name") + 1].lower()
                value = " ".join(parts[parts.index("value") + 1:]) if "value" in parts else ""
            except (ValueError, IndexError):
                continue

            if name == "netpath":
                if value and value not in ("<empty>", netpath):
                    search.stop_search()
                    try:
                        net = load_net(value)
                        netpath = value
                        search = Search(net, syzygy_path=current_syzygy_path)
                        print(f"info string Successfully loaded net from {value}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"info string Failed to load net from {value}: {e}", flush=True)

            elif name == "syzygypath":
                new_path = None if value in ("", "<empty>") else value
                if new_path != current_syzygy_path:
                    search.stop_search()
                    current_syzygy_path = new_path
                    search = Search(net, syzygy_path=new_path)

            elif name == "threads":
                # Conservé pour la compatibilité : le moteur est mono-thread par conception
                # (le MCTS en pur Python se parallélise mal à cause du GIL).
                try:
                    requested = int(value)
                except ValueError:
                    requested = 1
                if requested != 1:
                    print("info string DebZero searches on a single thread (Python GIL)", flush=True)

        elif cmd == "ucinewgame":
            search.stop_search()
            state_board = chess.Board()
            if search.evaluator is not None:
                search.evaluator.clear()

        elif cmd == "position":
            search.stop_search()
            state_board = position(command)

        elif cmd == "go":
            params = parse_go_params(command.split(" "))
            search.start(state_board, params)

        elif cmd == "stop":
            search.stop_search()

        elif cmd == "ponderhit":
            search.ponderhit()

        elif cmd in ("quit", "exit"):
            search.stop_search()
            break


if __name__ == "__main__":
    main()
