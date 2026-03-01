import onnxruntime as ort
import chess
from time import perf_counter
from src.utils.board_encoder import fen_to_planes_int8
import numpy as np
from src.utils.mapping_out_1858 import FlatMapper


def mirror_move(move: chess.Move) -> chess.Move:
    from_square = chess.square_mirror(move.from_square)
    to_square = chess.square_mirror(move.to_square)
    return chess.Move(from_square, to_square, promotion=move.promotion)

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

        return ort_session, input_name, target_dtype

    def _apply_temperature_and_normalize(self, move_probs, temp):
        """
        Applique la température dynamique donnée.
        """
        if temp == 1.0:
            return move_probs

        temp_probs = []
        sum_probs = 0.0
        inv_temp = 1.0 / temp

        for p, m in move_probs:
            # Sécurité mathématique
            if p > 1e-9:
                new_p = p ** inv_temp
            else:
                new_p = 0.0
            temp_probs.append((new_p, m))
            sum_probs += new_p

        if sum_probs == 0: return temp_probs

        # Renormalisation
        final_probs = [(p / sum_probs, m) for p, m in temp_probs]
        return final_probs

    def _get_dynamic_temperature(self, board: chess.Board) -> float:
        """
        Détermine la température selon le stade de la partie.
        """
        # Compte le nombre de pièces (Rois inclus)
        piece_count = board.occupied.bit_count()

        # --- MODE FINALE (ENDGAME) ---
        # Si 5 pièces ou moins (ex: Roi+Tour vs Roi, Roi+Pion vs Roi...)
        if piece_count <= 5:
            # On vérifie si on n'est pas en train de se faire mater (auquel cas on garde la précision)
            # Mais en général, on veut augmenter l'exploration pour trouver le mat.
            return 8.0  # Température EXTRÊME pour aplatir les probas et laisser l'heuristique décider

        # --- MODE MILIEU DE JEU ---
        return self.temperature

    def inference(self, board):

        if board.turn == chess.WHITE:
            board_to_encode = board
        else:
            board_to_encode = board.mirror()

        encoded_board = fen_to_planes_int8(board_to_encode.fen()).astype(self.target_type)
        encoded_board = np.expand_dims(encoded_board, 0)

        t_start = perf_counter()

        outputs = self.session.run(None, {self.input_name: encoded_board})

        t_end = perf_counter()

        output_0 = outputs[0]
        output_1 = outputs[1]

        policy_arr = output_0
        wdl_arr = output_1

        policy = policy_arr.reshape(1858)
        wdl = wdl_arr.reshape(3)

        move_list = list(board_to_encode.legal_moves)
        proba_moves = []

        for move in move_list:
            move_index = self.fm.get_move_index(move.uci())
            move_probability = policy[move_index]
            proba_moves.append((move_probability, move))

        # --- APPLICATION TEMPERATURE ---
        current_temp = self._get_dynamic_temperature(board)
        proba_moves = self._apply_temperature_and_normalize(proba_moves, current_temp)
        # -------------------------------

        if board.turn == chess.BLACK:
            proba_moves = [(p, mirror_move(m)) for (p, m) in proba_moves]

        return proba_moves, wdl, t_end - t_start

    def inference_batch(self, boards: list[chess.Board]) -> tuple:
        """
        Exécute l'inférence sur une liste de positions (batch).
        Renvoie : (liste_de_listes_moves_probs, liste_ de_wdls)
        """
        if not boards:
            return [], []

        batch_planes = []
        meta_data = []  # Pour stocker (work_board, is_black_turn)

        # 1. Prétraitement (Encoding)
        for board in boards:
            # Gestion de la symétrie : le réseau voit toujours comme les Blancs
            if board.turn == chess.WHITE:
                work_board = board
                is_black = False
            else:
                work_board = board.mirror()
                is_black = True

            # On stocke le board de travail pour générer les coups légaux plus tard
            meta_data.append((work_board, is_black))

            # Encodage FEN -> Tenseur
            encoded = fen_to_planes_int8(work_board.fen()).astype(self.target_type)
            batch_planes.append(encoded)

        # 2. Création du tenseur de batch (N, 13, 8, 8)
        # C'est ici que l'optimisation opère : np.stack est très efficace
        input_tensor = np.stack(batch_planes)

        # 3. Inférence groupée
        # ONNX Runtime va traiter les N positions simultanément
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Récupération des sorties brutes
        batch_policy_logits = outputs[0]  # Shape (N, 1858)
        batch_wdl = outputs[1]  # Shape (N, 3)

        results_policies = []
        results_wdls = []

        # 4. Post-traitement (Decoding)
        # On doit réassocier chaque sortie à ses coups légaux
        for i, (work_board, is_black) in enumerate(meta_data):

            # Extraction des données brutes pour l'élément i
            policy_arr = batch_policy_logits[i]  # Vecteur plat (1858,)
            wdl_arr = batch_wdl[i]  # Vecteur (3,)

            # Mapping : Index Réseau -> Coups Python
            move_list = list(work_board.legal_moves)
            proba_moves = []

            for move in move_list:
                move_index = self.fm.get_move_index(move.uci())
                move_probability = float(policy_arr[move_index])
                proba_moves.append((move_probability, move))

            # --- APPLICATION TEMPERATURE ---
            current_temp = self._get_dynamic_temperature(boards[i])
            proba_moves = self._apply_temperature_and_normalize(proba_moves, current_temp)
            # -------------------------------

            # Si c'était aux noirs, on remet les coups dans le bon sens (miroir inverse)
            if is_black:
                proba_moves = [(p, mirror_move(m)) for (p, m) in proba_moves]

            results_policies.append(proba_moves)
            results_wdls.append(wdl_arr)

        return results_policies, results_wdls


if __name__ == "__main__":
    fen = r"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    b = chess.Board(fen)
    net_path = r"..\..\nets\run_6_64_2_fp32.onnx"
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