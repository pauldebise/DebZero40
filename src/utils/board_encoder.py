import numpy as np
import chess

# Correspondance case -> (rangée, colonne) précalculée : évite 2 appels de fonction
# python-chess par pièce, ce qui représente un coût notable sur 32 pièces × 200 nœuds/s.
_RANK = [chess.square_rank(sq) for sq in range(64)]
_FILE = [chess.square_file(sq) for sq in range(64)]

# Signatures de pièces précalculées : (canal du trait, canal de l'adversaire) par type.
_CHANNEL = {}
for _pt in range(1, 7):
    _CHANNEL[(True, _pt)] = _pt - 1
    _CHANNEL[(False, _pt)] = _pt - 1 + 6

# Nombre de plans de pièces (historique, réseaux existants) et plans additionnels.
PIECE_PLANES = 12
EXTRA_PLANES = 5  # 4 droits de roque + 1 case de prise en passant
TOTAL_PLANES = PIECE_PLANES + EXTRA_PLANES


def board_to_planes_int8(board, extended: bool = False):
    """Encode une board en plans (8, 8, 12) ou (8, 8, 17) int8.

    Le plateau est vu du point de vue du joueur au trait : le réseau n'apprend que le
    côté blanc, et les plans sont retournés quand c'est aux noirs de jouer.

    `extended=True` ajoute 5 plans de constantes (canaux 12 à 16), toujours **du point de
    vue du joueur au trait** :

        12 : grand roque encore possible (dame)
        13 : petit roque encore possible (roi)
        14 : grand roque adverse encore possible
        15 : petit roque adverse encore possible
        16 : case cible de la prise en passant (un seul 1)

    Pourquoi c'est indispensable : sans ces plans, deux positions au plateau identique mais
    aux droits de roque différents produisent EXACTEMENT la même sortie réseau. Mesuré sur
    N1_10x64 : les roques ne reçoivent que 0,019 % de probabilité dans une position où ils
    sont les deux meilleurs coups possibles, alors qu'une prior neutre en donnerait 7,7 %.
    """
    channels = TOTAL_PLANES if extended else PIECE_PLANES
    planes = np.zeros((8, 8, channels), dtype=np.int8)

    black_turn = board.turn == chess.BLACK
    channel_of = _CHANNEL

    for square, piece in board.piece_map().items():
        rank = _RANK[square]
        if black_turn:
            rank = 7 - rank
        planes[rank, _FILE[square], channel_of[(piece.color == board.turn, piece.piece_type)]] = 1

    if not extended:
        return planes

    # --- droits de roque, exprimés du point de vue du joueur au trait
    # Note : côté visuel, la gauche est toujours « dame » et la droite toujours « roi »,
    # pour les deux couleurs. Le retournement des plans (noirs) ne change donc pas la
    # lecture de ces canaux : pas de permutation à faire.
    our_color = board.turn
    their_color = not board.turn

    if board.has_queenside_castling_rights(our_color):
        planes[:, :, 12] = 1
    if board.has_kingside_castling_rights(our_color):
        planes[:, :, 13] = 1
    if board.has_queenside_castling_rights(their_color):
        planes[:, :, 14] = 1
    if board.has_kingside_castling_rights(their_color):
        planes[:, :, 15] = 1

    if board.ep_square is not None:
        rank = _RANK[board.ep_square]
        if black_turn:
            rank = 7 - rank
        planes[rank, _FILE[board.ep_square], 16] = 1

    return planes


def fen_to_planes_int8(fen, extended: bool = False):
    # Conservé pour le pipeline d'entraînement (dataset_generator) et la compatibilité.
    return board_to_planes_int8(chess.Board(fen), extended=extended)
