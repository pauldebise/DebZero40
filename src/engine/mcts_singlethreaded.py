from typing import Optional, Dict, List, Tuple
import chess
import chess.polyglot
import chess.syzygy
import numpy as np
from src.utils.net_wrapper import NetWrapper
import threading
from time import perf_counter

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


class MctsNode:
    __slots__ = ["hash", "prior_prob", "visits", "total_wdl", "children", "nt"]

    def __init__(self, zobrist_hash: int = 0, prior_prob: float = 0.0, node_table: Optional[MctsTable] = None):
        self.hash = zobrist_hash
        self.total_wdl = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.children: Dict[chess.Move, int] = {}

        self.prior_prob = prior_prob
        self.visits = 0
        self.nt = node_table

    def is_leaf(self) -> bool:
        return not self.children

    def get_value(self) -> float:
        if not self.visits:
            return 0.0
        return (self.total_wdl[0] - self.total_wdl[2]) / self.visits

    def select_child(self) -> Tuple[Optional[chess.Move], Optional["MctsNode"]]:
        best_score = -float('inf')
        best_child_node = None
        best_move = None

        sqrt_visits = np.sqrt(self.visits)
        fpu_value = self.get_value() - 0.1

        for move, child_hash in self.children.items():
            child = self.nt.get_node(child_hash)

            if child.visits == 0:
                q = fpu_value
            else:
                q = -child.get_value()

            u = c_puct * child.prior_prob * sqrt_visits / (1 + child.visits)

            score = q + u

            if score > best_score:
                best_score = score
                best_child_node = child
                best_move = move

        return best_move, best_child_node


class MctsGraph:

    def __init__(self, net: NetWrapper, syzygy_path: str) -> None:
        self.node_table: MctsTable = MctsTable()
        self.root_board: chess.Board = chess.Board()
        self.root_node: Optional['MctsNode'] = None
        self.net: NetWrapper = net
        self.tb_hits = 0
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

    def expand_node(self, node: MctsNode, board: chess.Board, policy_prob: Dict[chess.Move, float]):
        legal_moves = list(board.legal_moves)
        total_prob = sum(policy_prob.get(m, 0.0) for m in legal_moves)

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

        visit_count = {}
        for move, zh in node.children.items():
            child = self.node_table.get_node(zh)
            visit_count[move] = child.visits

        total_visits = sum(visit_count.values())
        if total_visits > 0:
            visit_prob = {m: v / total_visits for m, v in visit_count.items()}
            best_move = max(visit_count.items(), key=lambda x: x[1])[0]
            return best_move, visit_prob

        return chess.Move.null(), {}

    def get_pv_line(self, node: MctsNode) -> List[chess.Move]:
        pv_line = []
        current = node
        seen = set()

        while not current.is_leaf():
            if current.hash in seen: break
            seen.add(current.hash)

            best_visits = -1
            best_move = None
            best_child = None

            for move, zh in current.children.items():
                child = self.node_table.get_node(zh)
                if child.visits > best_visits:
                    best_visits = child.visits
                    best_child = child
                    best_move = move

            if best_child is None or best_child.visits == 0:
                break

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

    def search(self, board: chess.Board, searchmove: List[chess.Move] = None, stop_event: threading.Event = None,
               time_manager=None, nodes=None):

        t_start = perf_counter()
        self.tb_hits = 0

        if searchmove is None:
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move, {}, [move]
                board.pop()

        root_hash = chess.polyglot.zobrist_hash(board)
        if self.node_table.is_node(root_hash):
            self.root_node = self.node_table.get_node(root_hash)
        else:
            self.root_node = MctsNode(zobrist_hash=root_hash, node_table=self.node_table)
            self.node_table.node_table[root_hash] = self.root_node

        if self.root_node.is_leaf():

            raw_policy, raw_wdl, _ = self.net.inference(board)
            policy_prob = {move: prob for prob, move in raw_policy}

            if nodes == 1:
                best_move = max(policy_prob, key=policy_prob.get)
                val = raw_wdl[0] - raw_wdl[2]
                if abs(val) < 0.99:

                    score_cp = int(-np.log((1.0 - val + 1e-5) / (1.0 + val + 1e-5)) * 300)
                    score_cp = max(min(score_cp, 2000), -2000)
                else:
                    score_cp = 10000 if val > 0 else -10000
                print(f"info depth 0 nodes 1 score cp {score_cp}")
                return best_move, {}, [best_move]

            if searchmove is None:
                moves_to_search = list(board.legal_moves)
            else:
                moves_to_search = [m for m in searchmove if m in board.legal_moves]
                if not moves_to_search: moves_to_search = list(board.legal_moves)

            if len(moves_to_search) > 1:
                noise = np.random.dirichlet([0.3] * len(moves_to_search))
                epsilon = 0.25
            else:
                noise = [0.0]
                epsilon = 0.0

            dict_policy_prob = {}
            for i, move in enumerate(moves_to_search):
                original_prob = policy_prob.get(move, 0.0)
                if epsilon > 0:
                    prob = (1 - epsilon) * original_prob + epsilon * noise[i]
                else:
                    prob = original_prob
                dict_policy_prob[move] = prob

            self.expand_node(self.root_node, board, dict_policy_prob)

        for node_count in range(nodes if (nodes is not None and nodes > 1) else 1_000_000_000):



            if node_count % 300 == 0 and node_count != 0:
                root_val = self.root_node.get_value()
                if abs(root_val) < 0.99:

                    score_cp = int(-np.log((1.0 - root_val + 1e-5) / (1.0 + root_val + 1e-5)) * 300)
                    score_cp = max(min(score_cp, 2000), -2000)
                else:
                    score_cp = 10000 if root_val > 0 else -10000

                search_duration = perf_counter() - t_start
                pv_line = self.get_pv_line(self.root_node)
                pv_str = " ".join([m.uci() for m in pv_line])

                nps = int(node_count // search_duration) if search_duration > 0 else 0
                seldepth = len(pv_line)
                depth = int((np.log(node_count) / 2) + 1)

                print(
                    f"info depth {depth} seldepth {seldepth} score cp {int(score_cp)} nodes {node_count} nps {nps} tbhits {self.tb_hits} time {int(1000 * search_duration)} pv {pv_str}",
                    flush=True)

            if stop_event and stop_event.is_set():
                break

            is_pondering = getattr(time_manager, 'is_pondering', False)
            max_time = getattr(time_manager, 'allocated_time', 5)
            current_timer_start = getattr(time_manager, 'timer_start', t_start)

            if not is_pondering and node_count > 0:
                if perf_counter() - current_timer_start >= max_time:
                    break

            node = self.root_node
            search_path = [node]
            steps = 0

            while not node.is_leaf() and not board.is_game_over():
                move, next_node = node.select_child()
                if next_node is None: break

                board.push(move)
                node = next_node
                search_path.append(node)
                steps += 1

                if board.is_repetition(2) or board.can_claim_draw():
                    break

            if board.is_game_over() or board.is_repetition(2) or board.is_insufficient_material():
                wdl = self.compute_terminal_wdl(board, len(search_path))
            else:

                syzygy_wdl = self.get_syzygy_wdl(board)

                if syzygy_wdl is not None:
                    self.tb_hits += 1
                    wdl = syzygy_wdl
                else:
                    policy_list, raw_wdl, _ = self.net.inference(board)
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

                    if node.is_leaf():
                        self.expand_node(node, board, policy_dict)

            for _ in range(len(search_path) - 1):
                board.pop()
            self.backpropagate(search_path, wdl)

        best_move, visit_prob = self.get_best_move(self.root_node)
        pv_line = self.get_pv_line(self.root_node)

        return best_move, visit_prob, pv_line


if __name__ == '__main__':
    net = NetWrapper(r"../../nets/test3/best_model_fp32.onnx")
    engine = MctsGraph(net, r"C:\Syzygy")
    board = chess.Board("8/4p3/1p1k4/8/3PP3/3K4/8/8 w - - 0 1")

    print(engine.search(board))
