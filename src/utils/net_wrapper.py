import onnxruntime as ort
import chess
from time import perf_counter
from src.utils.board_encoder import (
    board_to_planes_int8, fen_to_planes_int8, PIECE_PLANES, TOTAL_PLANES,
)
import numpy as np
from src.utils.mapping_out_1858 import FlatMapper
from src.utils.lc0_moves import LC0_MOVE_TO_INDEX


def mirror_move(move: chess.Move) -> chess.Move:
    from_square = chess.square_mirror(move.from_square)
    to_square = chess.square_mirror(move.to_square)
    return chess.Move(from_square, to_square, promotion=move.promotion)


# Cache des UCI miroir : calculé une seule fois par couple (from, to, promotion).
_MIRROR_UCI_CACHE = {}


def _mirror_uci(move: chess.Move) -> str:
    """UCI du coup vu depuis le côté noir (l'index 1858 ne connaît que les coups blancs)."""
    key = move
    cached = _MIRROR_UCI_CACHE.get(key)
    if cached is not None:
        return cached
    uci = mirror_move(move).uci()
    _MIRROR_UCI_CACHE[key] = uci
    return uci


def lc0_index_of(board: chess.Board, move: chess.Move) -> int | None:
    """Index du coup dans l'espace canonique LC0 (1858), ou ``None`` s'il n'y existe pas.

    Trois particularités par rapport à un simple ``move.uci()`` :

    * le roque est encodé comme « le roi prend sa propre tour » (``e1h1`` / ``e1a1``) ;
    * la sous-promotion en cavalier n'a pas d'entrée dédiée : LC0 utilise la case nue
      (``a7a8``), les sous-promotions dame/tour/fou ayant leurs propres entrées ;
    * la table ne contient que des coups blancs : les coups noirs sont miroir-retournés.
    """
    black = board.turn == chess.BLACK

    if board.is_castling(move):
        # `move` est roi -> case de destination ; la tour est sur le même rang.
        rank = chess.square_rank(move.from_square)
        rook_file = 7 if chess.square_file(move.to_square) == 6 else 0
        castling = chess.Move(move.from_square, chess.square(rook_file, rank))
        uci = _mirror_uci(castling) if black else castling.uci()
        return LC0_MOVE_TO_INDEX.get(uci)

    if move.promotion == chess.KNIGHT:
        base = chess.Move(move.from_square, move.to_square)
        uci = _mirror_uci(base) if black else base.uci()
        return LC0_MOVE_TO_INDEX.get(uci)

    uci = _mirror_uci(move) if black else move.uci()
    return LC0_MOVE_TO_INDEX.get(uci)


# --- table (départ, arrivée, promotion) -> index LC0, en coordonnées blanches -------
# `lc0_index_of` appelle `is_castling()` et construit une chaîne UCI pour chaque coup
# légal et chaque nœud : 30 à 40 chaînes par nœud, c'est le poste le plus coûteux de
# l'encodage. La table remplace tout ça par un accès dictionnaire.
_PROMO_OF = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}


def _build_move_index() -> dict:
    table = {}
    for uci, idx in LC0_MOVE_TO_INDEX.items():
        frm = chess.parse_square(uci[:2])
        to = chess.parse_square(uci[2:4])
        promo = _PROMO_OF[uci[4]] if len(uci) > 4 else None
        table[(frm, to, promo)] = idx

    # La sous-promotion en cavalier n'a pas d'entrée propre dans l'espace LC0 : elle
    # utilise la case nue (a7a8). On la rabat donc sur cette entrée.
    for (frm, to, promo), idx in list(table.items()):
        if promo in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            bare = table.get((frm, to, None))
            if bare is not None:
                table[(frm, to, chess.KNIGHT)] = bare
    return table


MOVE_INDEX = _build_move_index()


class NetWrapper:

    def __init__(self, net_path, cpu=True, temperature = 1.36):
        self.cpu = cpu
        self.session, self.input_name, self.target_type = self.initialize_session(net_path)
        self.fm = FlatMapper()
        self.temperature = temperature

    def initialize_session(self, net_path):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        if self.cpu:
            available_providers = ['CPUExecutionProvider']
        else:
            available_providers = [
            'DmlExecutionProvider',
            'CPUExecutionProvider'
        ]
        ort_session = ort.InferenceSession(net_path, sess_options=sess_options, providers=available_providers)
        input_meta = ort_session.get_inputs()[0]
        input_name = input_meta.name
        type_str = input_meta.type

        if 'float16' in type_str:
            target_dtype = np.float16
        else:
            target_dtype = np.float32

        # Nombre de plans attendus en entrée : 12 pour les réseaux existants, 17 pour les
        # réseaux réentraînés (droits de roque + compteur de 50 coups).
        self.input_planes = PIECE_PLANES
        shape = getattr(input_meta, "shape", None) or []
        if len(shape) == 4 and isinstance(shape[-1], int):
            if shape[-1] not in (PIECE_PLANES, TOTAL_PLANES):
                raise ValueError(
                    f"le réseau attend {shape[-1]} plans en entrée, "
                    f"seules les formes {PIECE_PLANES} et {TOTAL_PLANES} sont gérées"
                )
            self.input_planes = shape[-1]

        # Les nouveaux réseaux exportés par `src/train/export_onnx.py` demandent une
        # seconde entrée `legal_mask` (1858 valeurs) et sortent leur policy dans l'espace
        # canonique LC0. C'est le seul marqueur fiable de l'espace de politique : on ne
        # se fie PAS au nombre de plans (un réseau 17 plans pourrait réutiliser l'ancienne
        # liste maison).
        self.uses_legal_mask = False
        self.legal_mask_dtype = target_dtype
        for inp in ort_session.get_inputs():
            if getattr(inp, "name", None) == "legal_mask":
                self.uses_legal_mask = True
                if "float16" in (getattr(inp, "type", "") or ""):
                    self.legal_mask_dtype = np.float16
                else:
                    self.legal_mask_dtype = np.float32
                break

        # Espace de politique : LC0 dès qu'il y a un masque de légalité, sinon ancienne
        # liste maison (`FlatMapper`). Alias explicite pour la lisibilité du code moteur.
        self.uses_lc0 = self.uses_legal_mask

        return ort_session, input_name, target_dtype

    def _apply_temperature_and_normalize(self, move_probs, temp):
        if temp == 1.0:
            return move_probs

        temp_probs = []
        sum_probs = 0.0
        inv_temp = 1.0 / temp

        for p, m in move_probs:
            if p > 1e-9:
                new_p = p ** inv_temp
            else:
                new_p = 0.0
            temp_probs.append((new_p, m))
            sum_probs += new_p

        if sum_probs == 0: return temp_probs

        final_probs = [(p / sum_probs, m) for p, m in temp_probs]
        return final_probs

    def _get_dynamic_temperature(self, board: chess.Board) -> float:
        piece_count = board.occupied.bit_count()

        if piece_count <= 5:
            return 8.0

        return self.temperature

    def inference(self, board):
        """Policy (probabilité par coup) et WDL pour `board`, du point de vue du trait.

        Version rapide : pas de FEN intermédiaire, pas d'objet board miroir reconstruit.
        Le miroir noir est traité par l'encodage (case + canal) et par la conversion
        des coups, ce qui évite `board.mirror()` et ses allocations à chaque nœud.

        La température n'est plus appliquée ici : aplatir la policy à l'inférence dégrade
        la prior du MCTS. Le `temperature` du constructeur reste accepté mais n'est plus
        utilisé que par `inference_with_temperature` (usage auto-jeu).
        """
        black = board.turn == chess.BLACK

        encoded = board_to_planes_int8(board, extended=self.input_planes == TOTAL_PLANES)
        # Pas de miroir ni de rotation : `board_to_planes_int8` retourne déjà la vue du
        # côté au trait (il inverse la rangée quand c'est aux noirs de jouer), et les canaux
        # 0..5 / 6..11 sont rangés par « mes pièces / ses pièces ». C'est exactement ce que
        # faisait l'ancien chemin `board.mirror()` + FEN — vérifié par
        # bench/test_inference_equivalence.py (écart nul blancs ET noirs).
        encoded = encoded.astype(self.target_type, copy=False)[None]

        t_start = perf_counter()
        feed = {self.input_name: encoded}

        if self.uses_lc0:
            # Espace canonique LC0 : on calcule une seule fois l'index de chaque coup
            # légal, puis on s'en sert à la fois pour le masque et pour la lecture de la
            # policy. Seuls les coups légaux sont renvoyés au MCTS.
            indexed_moves = []
            legal_mask = np.zeros((1, 1858), dtype=self.legal_mask_dtype)
            king_sq = board.king(board.turn)
            mirror = 56 if black else 0
            for move in board.legal_moves:
                frm = move.from_square
                to = move.to_square
                if frm == king_sq and abs(chess.square_file(to) - chess.square_file(frm)) == 2:
                    # Roque : l'espace LC0 l'encode comme « le roi prend sa tour ».
                    to = chess.square(7 if chess.square_file(to) == 6 else 0,
                                      chess.square_rank(frm))
                idx = MOVE_INDEX.get((frm ^ mirror, to ^ mirror, move.promotion))
                if idx is None:
                    # Filet de sécurité : cas exotique absent de la table.
                    idx = lc0_index_of(board, move)
                if idx is None:
                    continue
                indexed_moves.append((move, idx))
                legal_mask[0, idx] = 1
            feed["legal_mask"] = legal_mask
        else:
            # Ancienne liste maison : comportement historique inchangé.
            indexed_moves = []
            for move in board.legal_moves:
                uci = move.uci() if not black else _mirror_uci(move)
                idx = self.fm.get_move_index(uci)
                if idx is not None:
                    indexed_moves.append((move, idx))

        outputs = self.session.run(None, feed)
        t_end = perf_counter()

        policy = outputs[0].reshape(1858)
        wdl = outputs[1].reshape(3)

        proba_moves = [(float(policy[idx]), move) for move, idx in indexed_moves]

        return proba_moves, wdl, t_end - t_start

    def inference_batch(self, boards: list[chess.Board]) -> tuple:

        if not boards:
            return [], []

        batch_planes = []
        batch_masks = [] if self.uses_legal_mask else None
        meta_data = []

        for board in boards:
            if board.turn == chess.WHITE:
                work_board = board
                is_black = False
            else:
                work_board = board.mirror()
                is_black = True

            meta_data.append((work_board, is_black))

            encoded = fen_to_planes_int8(
                work_board.fen(), extended=self.input_planes == TOTAL_PLANES
            ).astype(self.target_type)
            batch_planes.append(encoded)

            if batch_masks is not None:
                mask = np.zeros(1858, dtype=self.legal_mask_dtype)
                for move in work_board.legal_moves:
                    idx = lc0_index_of(work_board, move)
                    if idx is not None:
                        mask[idx] = 1
                batch_masks.append(mask)

        feed = {self.input_name: np.stack(batch_planes)}
        if batch_masks is not None:
            feed["legal_mask"] = np.stack(batch_masks)
        outputs = self.session.run(None, feed)

        batch_policy_logits = outputs[0]  # Shape (N, 1858)
        batch_wdl = outputs[1]  # Shape (N, 3)

        results_policies = []
        results_wdls = []

        for i, (work_board, is_black) in enumerate(meta_data):

            policy_arr = batch_policy_logits[i]
            wdl_arr = batch_wdl[i]

            move_list = list(work_board.legal_moves)
            proba_moves = []

            for move in move_list:
                if self.uses_lc0:
                    move_index = lc0_index_of(work_board, move)
                else:
                    move_index = self.fm.get_move_index(move.uci())
                if move_index is None:
                    continue
                move_probability = float(policy_arr[move_index])
                proba_moves.append((move_probability, move))

            current_temp = self._get_dynamic_temperature(boards[i])
            proba_moves = self._apply_temperature_and_normalize(proba_moves, current_temp)

            if is_black:
                proba_moves = [(p, mirror_move(m)) for (p, m) in proba_moves]

            results_policies.append(proba_moves)
            results_wdls.append(wdl_arr)

        return results_policies, results_wdls


if __name__ == "__main__":
    b = chess.Board()
    net_path = r"../../nets/N1_10x64/run_10_64_1_fp32.onnx"
    nw = NetWrapper(net_path, temperature=1)
    warmup = nw.inference(b)
    inference = nw.inference(b)
    print(inference)

    import matplotlib.pyplot as plt

    moves_data = inference[0]
    wdl_data = inference[1]

    probs = [item[0] for item in moves_data]
    move_names = [item[1].uci() for item in moves_data]

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = np.array(probs)[sorted_indices]
    sorted_moves = np.array(move_names)[sorted_indices]

    plt.figure()

    dt=0
    for _ in range(50):
        _, _, DT = nw.inference(b)
        dt += DT

    time_spent = 1000*dt/50

    bars = plt.bar(sorted_moves, sorted_probs, color='#4c72b0', edgecolor='black', alpha=0.8, label = f'{time_spent:.1f} ms')

    plt.legend()
    plt.ylabel('Probabilité (Policy)', fontsize=12)
    plt.xlabel('Coups Légaux', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    win_pct = wdl_data[0] * 100
    draw_pct = wdl_data[1] * 100
    loss_pct = wdl_data[2] * 100

    plt.title(f"Distribution de la Policy DebZero\nWin: {win_pct:.1f}% | Draw: {draw_pct:.1f}% | Loss: {loss_pct:.1f}%",
              fontsize=14, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        if height > 0.02:
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1%}',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    plt.show()