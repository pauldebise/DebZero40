import chess
import numpy as np

def fen_to_planes_int8(fen):

    board = chess.Board(fen)

    planes = np.zeros((8, 8, 12), dtype=np.int8)

    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        piece_offset = piece.piece_type - 1
        color_offset = 0 if piece.color == chess.WHITE else 6

        planes[rank, file, piece_offset + color_offset] = 1

    return planes

