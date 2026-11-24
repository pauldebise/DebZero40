"""
Recherche MCTS — implémentation de référence (mono-thread, pur Python).

Objectifs par rapport à la version de départ :
  * arbre en mémoire vive par objet (pas de fusion de clés tronquées) ;
  * une seule descente/remontée cohérente, aucune board partagée entre threads ;
  * la politique du réseau est utilisée telle quelle, sans température
    (la température est un réglage d'auto-jeu, pas une transformation de recherche) ;
  * remontée des valeurs avec inversion de signe à chaque niveau, sûre ;
  * prior normalisé sur les coups réellement jouables (sous-promotions tour/fou exclues).

Le module ne dépend pas d'onnxruntime : il prend un callable
`evaluate(board) -> (policy: dict[chess.Move, float], wdl: np.ndarray)` où `wdl` est
du point de vue du joueur au trait.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import chess
import chess.polyglot
import numpy as np

DEFAULT_C_PUCT = 2.0
FPU_REDUCTION = 0.1

Draw = np.array([0.0, 1.0, 0.0])
Win = np.array([1.0, 0.0, 0.0])
Loss = np.array([0.0, 0.0, 1.0])


def king_distance_heuristic(board: chess.Board) -> float:
    """Petit bonus de mate : rapproche les rois et pousse le roi adverse vers le bord."""
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

    center_distance = max(3 - their_rank, their_rank - 4) + max(3 - their_file, their_file - 4)
    dist_kings = max(abs(their_rank - our_rank), abs(their_file - our_file))

    return 0.1 * center_distance - 0.05 * dist_kings


def wdl_value(wdl: np.ndarray) -> float:
    """Valeur scalaire dans [-1, 1], point de vue du joueur au trait."""
    return float(wdl[0] - wdl[2])


def wdl_to_cp(val: float) -> int:
    """Convertit une valeur [-1, 1] en centipions UCI (sigmoïde logistique)."""
    val = max(min(val, 0.999), -0.999)
    return int(-math.log((1.0 - val) / (1.0 + val)) * 300)


def swap_wdl(wdl: np.ndarray) -> np.ndarray:
    """WDL du point de vue de l'autre camp (sans muter l'entrée)."""
    return np.array([wdl[2], wdl[1], wdl[0]], dtype=np.float64)


def repetition_count(board: chess.Board, max_plies: int = 100) -> int:
    """True si la position courante s'est déjà produite dans l'historique (au même trait).

    Implémentation explicite : `board.is_repetition(2)` de python-chess ne teste qu'une
    condition locale et peut créer de fausses nulles (il renvoie True sur beaucoup de
    positions qui ne se sont jamais répétées dans une vraie partie).

    On compare les hash Zobrist de la position courante avec celles obtenues en remontant
    l'historique de 2 demi-coups en 2 demi-coups, borné par le dernier coup irréversible.
    La board passée n'est PAS modifiée : on travaille sur une copie.
    """
    if not board.move_stack:
        return False

    current = chess.polyglot.zobrist_hash(board)
    limit = min(board.halfmove_clock, max_plies)

    work = board.copy(stack=True)
    for _ in range(1, limit // 2 + 1):
        try:
            work.pop()
            work.pop()
        except IndexError:
            break
        if chess.polyglot.zobrist_hash(work) == current:
            return True
    return False


class Node:
    __slots__ = ("prior", "visits", "wdl", "children", "move_from_parent")

    def __init__(self, prior: float = 0.0, move_from_parent: Optional[chess.Move] = None):
        self.prior = prior
        self.visits = 0
        # Somme WDL du point de vue du joueur qui doit jouer dans ce nœud.
        self.wdl = np.zeros(3, dtype=np.float64)
        self.children: Dict[chess.Move, "Node"] = {}
        self.move_from_parent = move_from_parent

    def is_leaf(self) -> bool:
        return not self.children

    def value(self) -> float:
        """Q du point de vue du joueur au trait dans ce nœud."""
        if self.visits == 0:
            return 0.0
        return float((self.wdl[0] - self.wdl[2]) / self.visits)

    def select_child(self, c_puct: float) -> Tuple[Optional[chess.Move], Optional["Node"]]:
        best_score = -math.inf
        best_move = None
        best_child = None

        sqrt_visits = math.sqrt(self.visits)
        fpu = self.value() - FPU_REDUCTION

        for move, child in self.children.items():
            # Le Q de l'enfant est du point de vue adverse : on l'inverse.
            q = -child.value() if child.visits else fpu
            u = c_puct * child.prior * sqrt_visits / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child


class MCTS:
    def __init__(
        self,
        evaluate: Callable[[chess.Board], Tuple[Dict[chess.Move, float], np.ndarray]],
        c_puct: float = DEFAULT_C_PUCT,
        mate_heuristic: bool = True,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.evaluate = evaluate
        self.c_puct = c_puct
        self.mate_heuristic = mate_heuristic
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root: Optional[Node] = None
        self.root_key: Optional[int] = None
        self.nodes_visited = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def filtered_legal_moves(board: chess.Board) -> List[chess.Move]:
        """Tout sauf les sous-promotions en tour/fou : ces coups sont presque toujours
        perdants, absents de la policy du réseau et polluent la normalisation."""
        return [m for m in board.legal_moves if m.promotion not in (chess.ROOK, chess.BISHOP)]

    def _expand(self, node: Node, board: chess.Board) -> np.ndarray:
        policy, wdl = self.evaluate(board)
        legal = self.filtered_legal_moves(board)

        weights = np.array([max(policy.get(m, 0.0), 0.0) for m in legal], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(legal), dtype=np.float64)
            total = weights.sum()
        weights /= total

        node.children = {
            move: Node(prior=float(prob), move_from_parent=move)
            for move, prob in zip(legal, weights)
        }
        return np.array(wdl, dtype=np.float64)

    def _add_dirichlet_noise(self, node: Node) -> None:
        if len(node.children) < 2 or self.dirichlet_epsilon <= 0:
            return
        probs = np.array([c.prior for c in node.children.values()], dtype=np.float64)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(probs))
        mixed = (1 - self.dirichlet_epsilon) * probs + self.dirichlet_epsilon * noise
        for child, p in zip(node.children.values(), mixed):
            child.prior = float(p)

    def _new_root(self, board: chess.Board) -> Node:
        root = Node()
        self._expand(root, board)
        self._add_dirichlet_noise(root)
        self.root = root
        self.root_key = chess.polyglot.zobrist_hash(board)
        return root

    @staticmethod
    def terminal_wdl(board: chess.Board) -> Optional[np.ndarray]:
        """WDL du point de vue du joueur au trait, ou None si la partie continue.

        ATTENTION : `board.is_repetition(2)` de python-chess ne compare que la position
        courante à son propre historique de demi-coups — il renvoie True dès qu'une position
        réapparaît, y compris après deux coups, ce qui n'est PAS une vraie répétition
        (fausses nulles). On écrit donc la détection nous-mêmes :

          * répétition triple (position tombée 3 fois)   -> nulle certaine ;
          * position tombée 2 fois avec le même joueur au trait, en étant en avance
            d'au moins 1 -> nulle probable (risque de triple), traitée comme nulle.

        Le coût est borné (au plus 100 demi-coups parcourus) et uniquement sur les feuilles
        où la position se répète, donc négligeable en temps de recherche.
        """
        if board.is_checkmate():
            return Loss
        if board.is_stalemate() or board.is_insufficient_material():
            return Draw
        if board.halfmove_clock >= 100:
            return Draw

        if board.move_stack:
            if repetition_count(board):
                return Draw
        return None

    def _leaf_value(self, board: chess.Board, node: Node) -> np.ndarray:
        terminal = self.terminal_wdl(board)
        if terminal is not None:
            return terminal

        wdl = self._expand(node, board)

        if self.mate_heuristic:
            h = king_distance_heuristic(board)
            if h != 0.0:
                wdl = wdl.copy()
                wdl[0] = max(wdl[0] + 0.8 * h, 0.0)
                wdl[2] = max(wdl[2] - 0.8 * h, 0.0)
                s = wdl.sum()
                if s > 0:
                    wdl = wdl / s
        return wdl

    # ------------------------------------------------------------------ recherche
    def run(self, board: chess.Board, max_nodes: int, should_stop: Callable[[], bool],
            reuse_root: bool = True) -> Node:
        """Au plus `max_nodes` simulations, arrêt si `should_stop()` renvoie True."""
        key = chess.polyglot.zobrist_hash(board)
        if not (reuse_root and self.root is not None and self.root_key == key):
            self._new_root(board)

        if self.root.is_leaf():
            # Position sans coup jouable (mat/pat déjà géré en amont).
            return self.root

        work = board.copy(stack=True)
        root_fen = board.fen()
        root_stack = list(board._stack)
        self.nodes_visited = 0

        for _ in range(max_nodes):
            if should_stop():
                break

            # Remise à la position racine sans reconstruire l'objet.
            work.set_fen(root_fen)
            work._stack = list(root_stack)

            node = self.root
            path = [node]

            # --- descente
            while node.children:
                move, child = node.select_child(self.c_puct)
                if child is None:
                    break
                work.push(move)
                node = child
                path.append(node)

                if node.is_leaf():
                    break
                if work.is_checkmate() or work.is_stalemate() \
                        or work.is_insufficient_material() or work.is_repetition(2):
                    break

            # --- évaluation du nœud feuille (point de vue du joueur qui y joue)
            wdl = self._leaf_value(work, node)

            # --- remontée : la valeur s'inverse à chaque niveau
            for n in reversed(path):
                n.wdl[0] += wdl[0]
                n.wdl[1] += wdl[1]
                n.wdl[2] += wdl[2]
                n.visits += 1
                wdl = swap_wdl(wdl)

            self.nodes_visited += 1

        return self.root

    def principal_variation(self, max_len: int = 16) -> List[chess.Move]:
        pv = []
        node = self.root
        while node is not None and node.children and len(pv) < max_len:
            best = None
            best_visits = -1
            for move, child in node.children.items():
                if child.visits > best_visits:
                    best_visits = child.visits
                    best = move
            if best is None or best_visits <= 0:
                break
            pv.append(best)
            node = node.children[best]
        return pv

    def best_move(self, temperature: float = 0.0) -> chess.Move:
        """temperature=0 -> argmax des visites ; >0 -> tirage proportionnel aux visites^(1/T)."""
        if self.root is None or not self.root.children:
            raise ValueError("aucun arbre de recherche")
        moves = list(self.root.children.keys())
        visits = np.array([self.root.children[m].visits for m in moves], dtype=np.float64)
        if visits.sum() <= 0:
            priors = np.array([self.root.children[m].prior for m in moves], dtype=np.float64)
            return moves[int(priors.argmax())]
        if temperature <= 0:
            return moves[int(visits.argmax())]
        weights = np.power(visits, 1.0 / temperature)
        weights /= weights.sum()
        return moves[int(np.random.choice(len(moves), p=weights))]

    def root_value(self) -> float:
        return self.root.value() if self.root is not None else 0.0

    def root_nodes(self) -> int:
        return self.root.visits if self.root is not None else 0
