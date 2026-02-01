import numpy as np
import chess

def fen_to_planes_int8(fen):
    board = chess.Board(fen)


    planes = np.zeros((8, 8, 12), dtype=np.int8)
    piece_map = board.piece_map()


    is_black_turn = (board.turn == chess.BLACK)

    for square, piece in piece_map.items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)


        if is_black_turn:
            rank = 7 - rank


        piece_type_idx = piece.piece_type - 1
        is_my_piece = (piece.color == board.turn)

        if is_my_piece:
            channel = piece_type_idx
        else:
            channel = piece_type_idx + 6

        planes[rank, file, channel] = 1

    return planes