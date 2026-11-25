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
EXTRA_PLANES = 5  # 4 droits de roque + 1 compteur de 50 coups normalisé
TOTAL_PLANES = PIECE_PLANES + EXTRA_PLANES


def board_to_planes_int8(board, extended: bool = False):
    """Encode une board en plans (8, 8, 12) ou (8, 8, 17).

    Le plateau est vu du point de vue du joueur au trait : le réseau n'apprend que le
    côté blanc, et les plans sont retournés quand c'est aux noirs de jouer.

    ``extended=False`` conserve **exactement** l'ancien encodage 12 plans (int8), pour
    rester compatible avec les réseaux déjà entraînés.

    ``extended=True`` produit les 17 plans des nouveaux réseaux (uint8, car le plan 16
    monte jusqu'à 255). Les canaux 12 à 16 sont, toujours **du point de vue du joueur
    au trait** :

        12 : grand roque encore possible (dame)
        13 : petit roque encore possible (roi)
        14 : grand roque adverse encore possible
        15 : petit roque adverse encore possible
        16 : `round(255 * min(halfmove_clock, 100) / 100)` sur toutes les cases

    Il n'y a plus de plan de prise en passant : elle était absente des données de
    distillation LC0 utilisées pour l'entraînement.

    Pourquoi les droits de roque sont indispensables : sans eux, deux positions au
    plateau identique mais aux droits différents produisent EXACTEMENT la même sortie
    réseau. Mesuré sur N1_10x64 : les roques ne reçoivent que 0,019 % de probabilité
    dans une position où ils sont les deux meilleurs coups possibles, alors qu'une
    prior neutre en donnerait 7,7 %.
    """
    channels = TOTAL_PLANES if extended else PIECE_PLANES
    dtype = np.uint8 if extended else np.int8
    planes = np.zeros((8, 8, channels), dtype=dtype)

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

    # --- compteur de 50 coups, normalisé sur 0..255 (même convention que l'entraînement)
    rule50 = min(int(board.halfmove_clock), 100)
    planes[:, :, 16] = int(round(255 * rule50 / 100))

    return planes


def fen_to_planes_int8(fen, extended: bool = False):
    # Conservé pour le pipeline d'entraînement (dataset_generator) et la compatibilité.
    return board_to_planes_int8(chess.Board(fen), extended=extended)
