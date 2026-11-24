import numpy as np
import chess

# Correspondance case -> (rangée, colonne) précalculée : évite 2 appels de fonction
# python-chess par pièce, ce qui représente un coût notable sur 32 pièces × 200 nœuds/s.
_RANK = [chess.square_rank(sq) for sq in range(64)]
_FILE = [chess.square_file(sq) for sq in range(64)]

# Signatures de pièces précalculées : (canal blanc, canal noir) par type et couleur.
_CHANNEL = {}
for _pt in range(1, 7):
    _CHANNEL[(True, _pt)] = _pt - 1
    _CHANNEL[(False, _pt)] = _pt - 1 + 6


def board_to_planes_int8(board):
    """Encode une board en plans (8, 8, 12) int8, sans passer par le FEN.

    Le plateau est vu du point de vue du joueur au trait (les noirs sont miroirés),
    comme dans le générateur de dataset : le réseau n'apprend que le côté blanc.
    """
    planes = np.zeros((8, 8, 12), dtype=np.int8)

    black_turn = board.turn == chess.BLACK
    channel_of = _CHANNEL

    # board.piece_map() renvoie un dict {case: Piece} ; on itère directement dessus.
    for square, piece in board.piece_map().items():
        rank = _RANK[square]
        if black_turn:
            rank = 7 - rank
        planes[rank, _FILE[square], channel_of[(piece.color == board.turn, piece.piece_type)]] = 1

    return planes


def fen_to_planes_int8(fen):
    # Conservé pour le pipeline d'entraînement (dataset_generator) et la compatibilité.
    return board_to_planes_int8(chess.Board(fen))
