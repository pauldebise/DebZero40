import sys
import os
import threading
from time import perf_counter
import chess
from src.engine.mcts_multithreaded import MctsGraph as MctsGraphMultiThreaded
from src.engine.mcts_singlethreaded import MctsGraph as MctsGraphSingleThreaded
from src.utils.net_wrapper import NetWrapper


def get_absolute_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def time_scheduler(side_to_move: bool, wtime=None, btime=None, winc=None, binc=None, movestogo=0, movetime=None, default=3, nodes=None):
    if movetime:
        return movetime / 1000.0

    if side_to_move == chess.WHITE:
        our_time = wtime if wtime is not None else 0
        our_inc = winc if winc is not None else 0
    else:
        our_time = btime if btime is not None else 0
        our_inc = binc if binc is not None else 0

    if our_time == 0:
        if nodes == 0:
            return default
        else:
            return float("inf")

    if movestogo > 0:
        t = our_time / movestogo + our_inc
    else:
        t = our_time / 20 + our_inc / 2

    t_sec = max(t / 1000 - 0.2, 0.1)
    return t_sec

def parse_go_params(command_list, board):
    params = {
        'wtime': None, 'btime': None, 'winc': 0, 'binc': 0,
        'movestogo': 0, 'movetime': None, 'ponder': False,
        'moves': [], 'nodes': None
    }

    it = iter(command_list[1:])
    for arg in it:
        if arg in params and arg != 'ponder' and arg != 'moves':
            params[arg] = int(next(it))
        elif arg == 'ponder':
            params['ponder'] = True
        elif arg == 'movetime':
            params['movetime'] = int(next(it))
        elif arg == 'searchmoves':
            search_moves = []
            try:
                while True:
                    move_uci = next(it)
                    search_moves.append(chess.Move.from_uci(move_uci))
            except StopIteration:
                pass
            params['searchmoves'] = search_moves

    return params


class TimeManager:
    def __init__(self):
        self.allocated_time = 1.0
        self.is_pondering = False
        self.timer_start = 0.0



class SearchManager:
    def __init__(self, engine):
        self.engine = engine
        self.search_thread = None
        self.stop_event = threading.Event()
        self.time_manager = TimeManager()
        self.saved_params = {}
        self.search_board = None

    def _calculate_and_set_time(self, params, board):
        allocated = time_scheduler(
            board.turn,
            wtime=params.get('wtime'),
            btime=params.get('btime'),
            winc=params.get('winc'),
            binc=params.get('binc'),
            movestogo=params.get('movestogo'),
            movetime=params.get('movetime'),
            nodes=params.get('nodes')
        )
        self.time_manager.allocated_time = allocated

    def start_search(self, board, params):

        self.stop_event.clear()
        self.search_board = board.copy()
        self.saved_params = params

        self.time_manager.timer_start = perf_counter()

        self.time_manager.is_pondering = params.get('ponder', False)

        if not self.time_manager.is_pondering:
            self._calculate_and_set_time(params, board)
        else:
            self.time_manager.allocated_time = float('inf')

        self.search_thread = threading.Thread(
            target=self._run_search,
            args=(self.search_board, params)
        )
        self.search_thread.start()

    def _run_search(self, board, params):

        best_move, _, pv_line = self.engine.search(
            board,
            searchmove=params.get('searchmoves', None),
            stop_event=self.stop_event,
            time_manager=self.time_manager,
            nodes=params.get('nodes', None)
        )


        if best_move is None:
            return

        ponder_move = pv_line[1] if len(pv_line) >= 2 else None

        res = f"bestmove {best_move.uci()}"
        if ponder_move:
            res += f" ponder {ponder_move.uci()}"

        print(res, flush=True)

    def stop_search(self):
        if self.search_thread and self.search_thread.is_alive():
            self.stop_event.set()
            self.search_thread.join()

    def ponderhit(self):
        if self.time_manager.is_pondering:
            self.time_manager.is_pondering = False
            self.time_manager.timer_start = perf_counter()
            self._calculate_and_set_time(self.saved_params, self.search_board)


def position(command_str: str) -> chess.Board:
    args = command_str.split()

    if not args:
        return chess.Board()

    if args[0] == 'position':
        args = args[1:]

    if args[0] == 'fen':
        if 'moves' in args:
            moves_idx = args.index('moves')
            fen_parts = args[1:moves_idx]
        else:
            fen_parts = args[1:]

        fen = " ".join(fen_parts)
        try:
            board = chess.Board(fen)
        except ValueError:
            board = chess.Board()

    elif args[0] == 'startpos':
        board = chess.Board()
    else:
        board = chess.Board()

    if 'moves' in args:
        moves_idx = args.index('moves')
        move_list = args[moves_idx + 1:]

        for move_uci in move_list:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    pass
            except ValueError:
                pass

    return board


def uci():
    print("id name DebZero40", flush=True)
    print("id author Paul Debise", flush=True)
    print("option name Ponder type check default false")
    print("option name Threads type spin default 16 min 1 max 128")
    print("option name NetPath type string default run_6_64_2_fp32.onnx")
    print("option name SyzygyPath type string default <empty>")
    print("uciok", flush=True)

def isready():
    print("readyok", flush=True)


if __name__ == '__main__':
    state_board = chess.Board()
    netpath = r"run_6_64_2_fp32.onnx"
    model_path = get_absolute_path(netpath)

    try:
        net = NetWrapper(model_path, cpu = True, temperature=1.5)
    except Exception as e:
        print("--------------------------------------------------")
        print(f""
              f"Fatal error encountered during start :")
        print(e)
        print("--------------------------------------------------")
        print(f"The engine searches for the nets here : {model_path}")
        print("Please check if nets are correctly placed.")
        print("The default net is run_6_64_2_fp32.onnx, it has to be correctly placed to start the engine.")
        print("--------------------------------------------------")
        input("Press enter to quit...")
        sys.exit(1)

    current_threads = 16
    current_syzygy_path = None


    def create_engine(network, thread_count, syzygy_path):
        if thread_count == 1:
            return MctsGraphSingleThreaded(network, syzygy_path=syzygy_path)
        else:
            return MctsGraphMultiThreaded(network, num_threads=thread_count, syzygy_path=syzygy_path)



    engine = create_engine(net, current_threads, current_syzygy_path)
    search_manager = SearchManager(engine)


    while True:
        try:
            command = input()
            if not command: continue
        except EOFError:
            break

        command_list = command.split(" ")
        cmd = command_list[0]

        if cmd == "uci":
            uci()
        elif cmd == "setoption":
            if 'name' in command_list and 'value' in command_list:
                try:
                    name_idx = command_list.index('name')
                    val_idx = command_list.index('value')

                    opt_name = command_list[name_idx + 1].lower()
                    opt_val = " ".join(command_list[val_idx + 1:])

                    if opt_name == "threads":
                        new_count = int(opt_val)
                        if new_count != current_threads:
                            search_manager.stop_search()
                            current_threads = new_count
                            engine = create_engine(net, current_threads, current_syzygy_path)
                            search_manager.engine = engine
                            if hasattr(engine, 'node_table'): engine.node_table.clear()
                            print(f"info string Engine switched to {current_threads} threads", flush=True)

                    elif opt_name == "syzygypath":
                        new_path = opt_val
                        if new_path == "<empty>": new_path = None

                        if new_path != current_syzygy_path:
                            search_manager.stop_search()
                            current_syzygy_path = new_path
                            engine = create_engine(net, current_threads, current_syzygy_path)
                            search_manager.engine = engine
                            if hasattr(engine, 'node_table'): engine.node_table.clear()
                            print(f"info string Syzygy path set to: {current_syzygy_path}", flush=True)

                    elif opt_name == "netpath":

                        new_path = opt_val
                        if new_path == "<empty>": new_path = None

                        if new_path != netpath:
                            search_manager.stop_search()

                            try:
                                netpath = new_path
                                model_path = get_absolute_path(netpath)
                                net = NetWrapper(model_path, cpu=True, temperature=1.5)
                                engine = create_engine(net, current_threads, current_syzygy_path)
                                print(f"info string Successfully loaded net from {netpath}", flush=True)
                            except Exception as e:
                                print(f"info string Failed to load net from {netpath}", flush=True)
                                print(f"info string Error : {e}", flush=True)

                        pass

                except ValueError:
                    pass

        elif cmd == "isready":
            isready()

        elif cmd == "ucinewgame":
            state_board = chess.Board()
            engine.root_node = None

        elif cmd == "position":
            search_manager.stop_search()
            state_board = position(command)
            engine.node_table.clear()
            engine.root = None

        elif cmd == "go":
            engine.node_table.clear()
            params = parse_go_params(command_list, state_board)
            search_manager.start_search(state_board, params)

        elif cmd == "stop":
            search_manager.stop_search()

        elif cmd == "ponderhit":
            search_manager.ponderhit()

        elif cmd in ["quit", "exit"]:
            search_manager.stop_search()
            break