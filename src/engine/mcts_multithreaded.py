import threading
from src.utils.net_wrapper import NetWrapper
import queue
import chess, chess.polyglot, chess.syzygy
import numpy as np
from typing import Optional, Tuple, Dict, List
import time


c_puct = 2.0


def king_distance_heuristic(board: chess.Board) -> float:
    if board.occupied.bit_count() > 3:
        return 0.0
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

    white_score = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in values.items())
    black_score = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in values.items())

    score_material = white_score - black_score

    if board.turn == chess.WHITE and score_material <= 5:
        return 0.0

    if board.turn == chess.BLACK and score_material >= -5:
        return 0.0

    our_king = board.king(board.turn)
    their_king = board.king(not board.turn)

    our_rank, our_file = chess.square_rank(our_king), chess.square_file(our_king)
    their_rank, their_file = chess.square_rank(their_king), chess.square_file(their_king)

    dist_center_rank = max(3 - their_rank, their_rank - 4)
    dist_center_file = max(3 - their_file, their_file - 4)
    center_distance = dist_center_rank + dist_center_file

    dist_kings = max(abs(their_rank - our_rank), abs(their_file - our_file))

    score = 0.1 * center_distance - 0.05 * dist_kings

    return score

class ChessBatcher:
    def __init__(self, net: NetWrapper, batch_size: int = 16):
        self.net = net
        self.batch_size = batch_size
        self.input_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            items = []
            try:
                items.append(self.input_queue.get(timeout=0.1))
            except queue.Empty:
                continue

            while len(items) < self.batch_size:
                try:
                    items.append(self.input_queue.get_nowait())
                except queue.Empty:
                    break

            boards = [item[0] for item in items]
            policies, wdls = self.net.inference_batch(boards)

            for i, item in enumerate(items):
                _, res_container, event = item
                res_container['policy'] = policies[i]
                res_container['wdl'] = wdls[i]
                event.set()

    def fetch_inference(self, board: chess.Board):
        event = threading.Event()
        res_container = {}
        self.input_queue.put((board.copy(), res_container, event))
        event.wait()
        return res_container['policy'], res_container['wdl']


class MctsNode:
    __slots__ = ["hash", "prior_prob", "visits", "total_wdl", "children", "nt", "vloss"]

    def __init__(self, zobrist_hash: int, prior_prob: float, node_table: 'MctsTable'):
        self.hash = zobrist_hash
        self.total_wdl = np.zeros(3, dtype=np.float64)
        self.children = {}
        self.prior_prob = prior_prob
        self.visits = 0
        self.vloss = 0
        self.nt = node_table

    def get_value(self) -> float:
        v_visits = self.visits + self.vloss
        if v_visits == 0: return 0.0
        return ((self.total_wdl[0] + self.vloss) - self.total_wdl[2]) / v_visits

    def is_leaf(self) -> bool:
        return not self.children

    def select_child(self) -> Tuple[Optional[chess.Move], Optional["MctsNode"]]:
        best_score = -float('inf')
        best_child_node = None
        best_move = None

        sqrt_visits = np.sqrt(self.visits + self.vloss)
        fpu_value = self.get_value() - 0.1

        for move, child_hash in self.children.items():
            child = self.nt.get_node(child_hash)

            if child.visits == 0:
                q = fpu_value
            else:
                q = -child.get_value()

            u = c_puct * child.prior_prob * sqrt_visits / (1 + child.visits + child.vloss)

            score = q + u

            if score > best_score:
                best_score = score
                best_child_node = child
                best_move = move

        return best_move, best_child_node


class MctsTable:

    __slots__ = ["node_table"]

    def __init__(self):
        self.node_table: Dict[int, MctsNode] = {}

    def clear(self) -> None:
        self.node_table.clear()

    def get_node(self, zobrist_hash: int) -> "MctsNode":
        return self.node_table[zobrist_hash]

    def is_node(self, zobrist_hash: int) -> bool:
        return zobrist_hash in self.node_table

    def create_node(self, zobrist_hash: int, prior_prob: float, node_table: 'MctsTable'):
        self.node_table[zobrist_hash] = MctsNode(
            zobrist_hash=zobrist_hash,
            prior_prob=prior_prob,
            node_table=node_table
        )

    def __len__(self) -> int:
        return len(self.node_table)


class MctsGraph:

    def __init__(self, net, num_threads: int = 8, syzygy_path: str = None):
        self.root_node = None
        self.node_table = MctsTable()
        self.net = net
        self.batcher = ChessBatcher(net, batch_size=num_threads)
        self.num_threads = num_threads
        self.lock = threading.Lock()
        self.tb_hits = 0
        self.node_counter=0

        self.tablebase = None
        if syzygy_path:
            try:
                self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
                print(f"info string Syzygy Tablebases found at {syzygy_path}", flush=True)
            except Exception:
                pass
                print("info string Warning: Syzygy path not valid", flush=True)

    def get_syzygy_wdl(self, board):
        if self.tablebase is None: return None
        if board.occupied.bit_count() > 5: return None

        try:
            wdl_score = self.tablebase.get_wdl(board)
            if wdl_score is None: return None

            if wdl_score > 1:
                return np.array([1.0, 0.0, 0.0])
            elif wdl_score < -1:
                return np.array([0.0, 0.0, 1.0])
            else:
                return np.array([0.0, 1.0, 0.0])
        except Exception:
            return None

    def get_best_move_from_syzygy(self, board: chess.Board) -> Optional[chess.Move]:
        if self.tablebase is None: return None

        try:
            root_wdl = self.tablebase.get_wdl(board)
        except:
            return None

        if root_wdl is None: return None

        best_move = None

        best_score = (-float('inf'), -float('inf'), -float('inf'))

        import random
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)

        for move in legal_moves:

            is_zeroing = (board.is_capture(move) or
                          move.promotion or
                          board.piece_type_at(move.from_square) == chess.PAWN)

            if board.halfmove_clock >= 99 and not is_zeroing:
                continue

            board.push(move)

            if root_wdl > 0 and board.is_repetition(2):
                board.pop()
                continue

            try:
                wdl_opp = self.tablebase.get_wdl(board)
                dtz_opp = self.tablebase.get_dtz(board)

                if wdl_opp is not None and dtz_opp is not None:

                    if wdl_opp < 0:
                        my_wdl = 1
                    elif wdl_opp > 0:
                        my_wdl = -1
                    else:
                        my_wdl = 0

                    abs_dtz = abs(dtz_opp)

                    score_zeroing = 0
                    if my_wdl > 0 and is_zeroing:
                        score_zeroing = 1

                    if my_wdl > 0:
                        score_dtz = -abs_dtz
                    elif my_wdl < 0:
                        score_dtz = abs_dtz
                    else:
                        score_dtz = 0

                    current_score = (my_wdl, score_zeroing, score_dtz)

                    if current_score > best_score:
                        best_score = current_score
                        best_move = move

            except Exception:
                pass

            board.pop()

        return best_move

    def expand_node(self, node: MctsNode, board: chess.Board, policy_prob: Dict[chess.Move, float]):
        legal_moves = list(board.legal_moves)

        filtered_moves = []
        for m in legal_moves:
            if m.promotion:
                if m.promotion == chess.QUEEN or m.promotion == chess.KNIGHT:
                    filtered_moves.append(m)
            else:
                filtered_moves.append(m)
        legal_moves = filtered_moves

        total_prob = sum(policy_prob.get(m, 0.0) for m in legal_moves)
        if total_prob == 0: total_prob = 1

        for move in legal_moves:
            prob = policy_prob.get(move, 0.0) / total_prob
            board.push(move)
            z_hash = chess.polyglot.zobrist_hash(board)
            board.pop()
            node.children[move] = z_hash
            if not self.node_table.is_node(z_hash):
                self.node_table.create_node(z_hash, prob, self.node_table)

    def backpropagate(self, search_path: List[MctsNode], wdl: np.ndarray) -> None:
        for node in search_path[::-1]:
            node.total_wdl += wdl
            node.visits += 1
            wdl[0], wdl[2] = wdl[2], wdl[0]

    def get_best_move(self, node: MctsNode) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        if not node.children: return chess.Move.null(), {}

        best_move = max(node.children.items(), key=lambda item: self.node_table.get_node(item[1]).visits)[0]
        return best_move, {}

    def get_pv_line(self, node: MctsNode) -> List[chess.Move]:
        pv_line = []
        current = node
        seen = set()
        while not current.is_leaf():
            if current.hash in seen: break
            seen.add(current.hash)

            best_move = None
            best_child = None
            best_visits = -1

            for m, h in current.children.items():
                c = self.node_table.get_node(h)
                if c.visits > best_visits:
                    best_visits = c.visits
                    best_child = c
                    best_move = m

            if best_child is None or best_child.visits == 0: break
            pv_line.append(best_move)
            current = best_child
        return pv_line

    def compute_terminal_wdl(self, board: chess.Board, depth: int) -> np.ndarray:
        if board.is_repetition(2) or board.is_stalemate() or board.is_insufficient_material():
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)

        result = board.result()
        if result == "1-0":
            winner = 1
        elif result == "0-1":
            winner = -1
        else:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)

        current_player = 1 if board.turn == chess.WHITE else -1
        if current_player * winner > 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _worker_thread(self, root_board: chess.Board, stop_event: threading.Event, max_nodes):
        max_nodes = max_nodes if max_nodes is not None else float("inf")
        while not stop_event.is_set() and self.node_counter < max_nodes:
            board = root_board.copy()
            node = self.root_node
            search_path = [node]

            with self.lock:
                while not node.is_leaf() and not board.is_game_over():
                    move, next_node = node.select_child()
                    if next_node is None: break
                    board.push(move)
                    node = next_node
                    node.vloss += 1
                    search_path.append(node)

            if board.is_game_over() or board.is_repetition(2) or board.is_insufficient_material():
                wdl = self.compute_terminal_wdl(board, len(search_path))
            else:
                syzygy_wdl = self.get_syzygy_wdl(board)

                if syzygy_wdl is not None:
                    with self.lock:
                        self.tb_hits += 1
                    wdl = syzygy_wdl
                else:
                    policy_list, raw_wdl = self.batcher.fetch_inference(board)
                    policy_dict = {m: p for p, m in policy_list}
                    wdl = np.array(raw_wdl, dtype=np.float64)

                    heuristic_score = king_distance_heuristic(board)
                    if heuristic_score != 0:
                        w_heuristic = 0.8
                        wdl[0] += heuristic_score * w_heuristic
                        wdl[2] -= heuristic_score * w_heuristic
                        wdl = np.maximum(wdl, 0.0)
                        s = np.sum(wdl)
                        if s > 0: wdl /= s

                    with self.lock:
                        if node.is_leaf():
                            self.expand_node(node, board, policy_dict)

            with self.lock:
                for n in search_path[1:]:
                    n.vloss -= 1
                self.backpropagate(search_path, np.array(wdl))
                self.node_counter += 1

    def search(self, board: chess.Board, searchmove: List[chess.Move] = None, stop_event: threading.Event = None,
               time_manager=None, nodes=None):

        self.tb_hits = 0
        self.node_counter = 0

        if searchmove is None:
            for move in board.legal_moves:
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move, {}, [move]
                board.pop()

        root_hash = chess.polyglot.zobrist_hash(board)
        if not self.node_table.is_node(root_hash):
            self.node_table.create_node(root_hash, 0.0, self.node_table)
        self.root_node = self.node_table.get_node(root_hash)

        if self.root_node.is_leaf():
            p, raw_wdl = self.batcher.fetch_inference(board)
            policy_dict = {m: prob for prob, m in p}

            if nodes == 1:
                best_move = max(policy_dict, key=policy_dict.get)
                val = raw_wdl[0] - raw_wdl[2]
                if abs(val) < 0.99:

                    score_cp = int(-np.log((1.0 - val + 1e-5) / (1.0 + val + 1e-5)) * 300)
                    score_cp = max(min(score_cp, 2000), -2000)
                else:
                    score_cp = 10000 if val > 0 else -10000
                print(f"info depth 0 nodes 1 score cp {score_cp}")
                return best_move, {}, [best_move]

            moves = list(policy_dict.keys())
            if len(moves) > 1:
                noise = np.random.dirichlet([0.3] * len(moves))
                epsilon = 0.25
                for i, move in enumerate(moves):
                    policy_dict[move] = (1 - epsilon) * policy_dict[move] + epsilon * noise[i]

            if searchmove:
                filtered = {m: policy_dict[m] for m in searchmove if m in policy_dict}
                if filtered:
                    s = sum(filtered.values())
                    policy_dict = {m: v / s for m, v in filtered.items()}

            self.expand_node(self.root_node, board, policy_dict)

        internal_stop = threading.Event()
        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self._worker_thread, args=(board, internal_stop, nodes))
            t.start()
            threads.append(t)

        start_time = time.perf_counter()
        last_print = start_time

        try:
            while True:
                if stop_event and stop_event.is_set(): break
                now = time.perf_counter()

                if nodes is not None and self.node_counter >= nodes:
                    break

                if time_manager and not time_manager.is_pondering:
                    if now - time_manager.timer_start >= time_manager.allocated_time: break

                elif time_manager is None and nodes is None and now - start_time > 5.0:
                    break

                if now - last_print > 0.5:
                    self._print_search_info(start_time)
                    last_print = now
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

        internal_stop.set()
        for t in threads: t.join()

        self._print_search_info(start_time)
        best_move, _ = self.get_best_move(self.root_node)
        pv_line = self.get_pv_line(self.root_node)

        if board.occupied.bit_count() <= 5 and self.tablebase is not None:
            syzygy_best = self.get_best_move_from_syzygy(board)
            if syzygy_best:
                print(f"info string Syzygy Perfect Move: {syzygy_best.uci()}", flush=True)
                return syzygy_best, {}, [syzygy_best]

        return best_move, {}, pv_line

    def _print_search_info(self, start_time):
        visits = self.root_node.visits
        if visits == 0: return
        duration = max(time.perf_counter() - start_time, 0.001)
        nps = int(visits / duration)
        q_val = max(min(self.root_node.get_value(), 0.99), -0.99)
        cp = int(-np.log((1.0 - q_val) / (1.0 + q_val)) * 200)
        pv_moves = self.get_pv_line(self.root_node)
        pv_str = " ".join([m.uci() for m in pv_moves])
        seldepth = len(pv_moves)
        depth  = int((np.log(len(self.node_table))/2)+1)
        print(
            f"info depth {depth} seldepth {seldepth} score cp {cp} nodes {visits} nps {nps} tbhits {self.tb_hits} time {int(duration * 1000)} pv {pv_str}",
            flush=True)
if __name__ == '__main__':
    cpu = False
    net = NetWrapper(r"../../nets/test3/best_model_fp32.onnx", cpu=cpu)
    engine = MctsGraph(net, num_threads=128, syzygy_path=r"C:\Syzygy")
    board = chess.Board()

    print(engine.search(board))
